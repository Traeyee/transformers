#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyi@zuoshouyisheng.com
# Created Time: 16 May 2020 19:17
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    get_linear_schedule_with_warmup,
    BertTokenizer,
)

from sentence_similarity_utilization import (
    AlbertForSiameseSimilaritySimple,
    SentencePairExample,
    )

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def convert_examples_to_dataset(examples, evaluate):
    from torch.utils.data import TensorDataset
    input_ids1 = torch.cat([_eg.sentence1.input_ids for _eg in examples])
    attention_mask1 = torch.cat(
        [_eg.sentence1.attention_mask for _eg in examples])
    token_type_ids1 = torch.cat(
        [_eg.sentence1.token_type_ids for _eg in examples])
    input_ids2 = torch.cat([_eg.sentence2.input_ids for _eg in examples])
    attention_mask2 = torch.cat(
        [_eg.sentence2.attention_mask for _eg in examples])
    token_type_ids2 = torch.cat(
        [_eg.sentence2.token_type_ids for _eg in examples])
    return TensorDataset(
        input_ids1,
        attention_mask1,
        token_type_ids1,
        input_ids2,
        attention_mask2,
        token_type_ids2,
        torch.tensor([_eg.label for _eg in examples], dtype=torch.float),
    )


def get_examples(csv_path, tokenizer, evaluate):
    import csv
    csv_reader = csv.reader(open(csv_path, "r"))
    examples = []
    for cols in csv_reader:
        if evaluate:
            if "id" != cols[0]:
                sntn_pair_eg = SentencePairExample(cols[1], cols[2], tokenizer)
                examples.append(sntn_pair_eg)
        else:
            if cols[2] in {"0", "1"}:
                sntn_pair_eg = SentencePairExample(cols[0], cols[1], tokenizer,
                                                   cols[2])
                examples.append(sntn_pair_eg)
    return examples


def load_and_cache_examples(args,
                            tokenizer,
                            evaluate=False,
                            output_examples=False):
    # Load data features from cache or dataset file
    input_dir = "TEST_SS"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    cached_features_file = os.path.join(
        args.output_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            list(filter(None, input_dir.split("/"))).pop(),
        ),
    )
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and args.read_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        dataset, examples = (
            # features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if evaluate:
            examples = get_examples(args.predict_file,
                                    tokenizer,
                                    evaluate=False)
        else:
            examples = get_examples(args.train_file, tokenizer, evaluate=False)
        dataset = convert_examples_to_dataset(examples, evaluate)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(
                {
                    # "features": features,
                    "dataset": dataset,
                    "examples": examples
                },
                cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples
    return dataset


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(
            train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    t_total = len(train_dataloader
                  ) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(
            args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split(
                "/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) //
                                             args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d",
                        global_step)
            logger.info("  Will skip the first %d steps in the first epoch",
                        steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            input_ids1 = batch[0]
            attention_mask1 = batch[1]
            token_type_ids1 = batch[2]
            input_ids2 = batch[3]
            attention_mask2 = batch[4]
            token_type_ids2 = batch[5]
            label = batch[6]
            inputs = {
                "input_ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "token_type_ids1": token_type_ids1,
                "position_ids1": None,
                "head_mask1": None,
                "inputs_embeds1": None,
                "input_ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "token_type_ids2": token_type_ids2,
                "position_ids2": None,
                "head_mask2": None,
                "inputs_embeds2": None,
                "label": label,
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean(
                )  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [
                        -1, 0
                ] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value,
                                                 global_step)
                    tb_writer.add_scalar("lr",
                                         scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) /
                                         args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [
                        -1, 0
                ] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args,
                               os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(),
                               os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(),
                               os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s",
                                output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples = load_and_cache_examples(args,
                                                tokenizer,
                                                evaluate=True,
                                                output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            input_ids1 = batch[0]
            attention_mask1 = batch[1]
            token_type_ids1 = batch[2]
            input_ids2 = batch[3]
            attention_mask2 = batch[4]
            token_type_ids2 = batch[5]
            inputs = {
                "input_ids1": input_ids1,
                "attention_mask1": attention_mask1,
                "token_type_ids1": token_type_ids1,
                "position_ids1": None,
                "head_mask1": None,
                "inputs_embeds1": None,
                "input_ids2": input_ids2,
                "attention_mask2": attention_mask2,
                "token_type_ids2": token_type_ids2,
                "position_ids2": None,
                "head_mask2": None,
                "inputs_embeds2": None,
                "label": None,
            }

            outputs = model(**inputs)
            label = batch[6]
            all_results.append({"pred": outputs, "label": label})

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)",
                evalTime, evalTime / len(dataset))

    # Compute the F1 and exact scores.
    import numpy as np
    import collections
    results = collections.OrderedDict()
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        TP, TN, FP, FN = 0, 0, 0, 0
        loss = 0.0
        for result in all_results:
            pred_bool = result["pred"] > threshold
            label_bool = result["label"] > 0.5  # 0 or 1
            for i in range(pred_bool.shape[0]):
                pred = pred_bool[i].item()
                label = label_bool[i].item()
                if pred:
                    if label:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if label:
                        FN += 1
                    else:
                        TN += 1
            loss += np.square(result["pred"].cpu().numpy() -
                              result["label"].cpu().numpy()).mean()
        loss /= TP + FP + FN + TN
        recall = float(TP) / (TP + FN + 1e-8)
        precision = float(TP) / (TP + FP + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        results["recall_{}".format(threshold)] = recall * 100
        results["precision_{}".format(threshold)] = precision * 100
        results["f1_{}".format(threshold)] = f1 * 100
        results["loss"] = loss
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help=
        "The input training file. If a data dir is specified, will look for the file there"
        +
        "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help=
        "The input evaluation file. If a data dir is specified, will look for the file there"
        +
        "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--read_cache",
        action="store_true",
        help="Read the cached training and evaluation sets")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
    )
    model = AlbertForSiameseSimilaritySimple.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=None,
    )
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args,
                                                tokenizer,
                                                evaluate=False,
                                                output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)
    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1
                          or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AlbertForSiameseSimilaritySimple.from_pretrained(
            args.output_dir)  # , force_download=True)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model.to(args.device)

        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info(
                "Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                import glob
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(
                        glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME,
                                  recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(
                    logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation",
                        args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            model = AlbertForSiameseSimilaritySimple.from_pretrained(
                checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict(
                (k + ("_{}".format(global_step) if global_step else ""), v)
                for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == '__main__':
    main()
