# This script is based on https://github.com/salesforce/LAVIS/blob/main/lavis/runners/runner_base.py

import datetime
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from NatureLM.config import Config
from NatureLM.dist_utils import get_rank, get_world_size, is_dist_avail_and_initialized, is_main_process, main_process
from NatureLM.logger import MetricLogger, SmoothedValue
from NatureLM.optims import LinearWarmupCosineLRScheduler, get_optimizer
from NatureLM.task_metrics import get_task_metrics
from NatureLM.utils import get_dataloader, prepare_sample_dist


class Runner:
    def __init__(self, cfg: Config, model, datasets, job_id):
        self.config = cfg

        # log
        device = "cuda:0"
        if is_main_process():
            if self.config.run.wandb_enabled:
                wandb.init(project="earthlm", config=self.config.model_dump())
            else:
                wandb.init(mode="disabled")

        if "LOCAL_RANK" in os.environ:
            device = int(os.environ["LOCAL_RANK"])
        else:
            device = self.config.run.device
        print(f"device is {device} could have been {self.config.run.device}")
        self.output_dir = Path(self.config.run.output_dir) / job_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_writter = SummaryWriter(self.output_dir)

        # settings
        self.device = torch.device(device)
        self.use_distributed = self.config.run.use_distributed
        self.start_epoch = 0
        self.max_epoch = self.config.run.optims.max_epoch
        self.evaluate_only = self.config.run.evaluate
        self.cuda_enabled = self.device.type == "cuda"

        # test prompt
        self.prompt_template = self.config.model.prompt_template

        # model
        self._model = model
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
        self._model.to(self.device)
        if self.use_distributed:
            self.model = DDP(
                self._model,
                find_unused_parameters=True,
                static_graph=False,
                device_ids=[self.device],
            )
        else:
            self.model = self._model

        # dataloaders
        self.train_loader = get_dataloader(
            datasets["train"],
            self.config.run,
            is_train=True,
            use_distributed=self.use_distributed,
        )
        self.valid_loader = get_dataloader(
            datasets["valid"],
            self.config.run,
            is_train=False,
            use_distributed=self.use_distributed,
        )
        self.test_loader = get_dataloader(
            datasets["test"],
            self.config.run,
            is_train=False,
            use_distributed=self.use_distributed,
        )

        # scaler
        self.use_amp = self.config.run.amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # optimizer & scheduler
        self.iters_per_epoch = (
            len(self.train_loader) if self.config.run.epoch_based else self.config.run.iters_per_epoch
        )
        self.optimizer = get_optimizer(self.model, self.config.run.optims)
        self.scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            max_epoch=self.max_epoch,
            iters_per_epoch=self.iters_per_epoch,
            min_lr=self.config.run.optims.min_lr,
            init_lr=self.config.run.optims.init_lr,
            warmup_steps=self.config.run.optims.warmup_steps,
            warmup_start_lr=self.config.run.optims.warmup_start_lr,
        )

        #### augmentations
        # self.rng = random.Random(self.config.run.seed)
        # self.rngnp = np.random.default_rng(seed=self.config.run.seed)
        # self.rngth = torch.Generator(device=args.device)
        # self.rngth.manual_seed(self.config.run.seed)
        # augments = []
        # if self.config.run.augmentations.flip:
        #     augments.append(augmentations.Flip(self.config.run.augmentations.flip, rngth=self.rngth, seed=self.config.run.seed))
        # if self.config.run.augmentations.bandmask:
        #     augments.append(augmentations.BandMask(self.config.run.augmentations.bandmask, sample_rate=args.sample_rate, rng=self.rng, seed=self.config.run.seed))
        # if self.config.run.augmentations.revecho:
        #     augments.append(
        #         augmentations.RevEcho(proba=self.config.run.augmentations.revecho,rng=self.rng,seed=self.config.run.seed))
        # self.augment = torch.nn.Sequential(*augments)

        self.log_config()

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def train_epoch(self, epoch):
        self.model.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        logging.info("Start training epoch {}, {} iters per inner epoch.".format(epoch, self.iters_per_epoch))
        header = "Train: data epoch: [{}]".format(epoch)

        # Get gradient clipping parameters from config
        clip_grad_norm = self.config.run.optims.max_grad_norm
        clip_grad_value = self.config.run.optims.max_grad_value

        for i in metric_logger.log_every(
            range(self.iters_per_epoch),
            self.config.run.log_freq,
            header=header,
            logger=self.log_writter,
            start_step=epoch * self.iters_per_epoch,
        ):
            if i >= self.iters_per_epoch:
                break

            samples = next(self.train_loader)

            samples = prepare_sample_dist(samples, self.device)

            #### augmentation
            # if False:
            #     samples = self.augment(samples)

            self.scheduler.step(cur_epoch=epoch, cur_step=i)

            with torch.autocast(self.device.type, enabled=self.use_amp, dtype=torch.bfloat16):
                loss = self.model(samples)["loss"]
                if torch.isnan(loss):
                    print("loss nan", samples)
                #     continue

            if self.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Apply gradient clipping
            if clip_grad_norm is not None:
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad_norm)
            if clip_grad_value is not None:
                if self.use_amp and self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_grad_value)

            if (i + 1) % self.config.run.accum_grad_iters == 0:
                if self.use_amp and self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

        metric_logger.synchronize_between_processes()
        logging.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def valid_epoch(self, epoch, split, decode=True, save_json=False, decode_ratio=1.0):
        """
        Decode = True will lead to calculation of custom metrics which are based on text.
        decode_ratio controls the percentage of batches which will have custom metrics computed,
        a speed trade-off due to the cost of the 'generate' method.
        """
        model = self.unwrap_dist_model(self.model)
        model.eval()

        dataloader = getattr(self, split + "_loader", None)
        assert dataloader is not None, f"{split}_loader does not exist."

        metric_logger = MetricLogger(delimiter="  ")
        header = f"Eval: data epoch: [{epoch}]"

        results_per_task = defaultdict(list)  # Store results per task
        overall_results = []  # Store all results for overall metrics

        # Calculate N based on decode_ratio
        if decode_ratio <= 0.0:
            N = float("inf")  # Effectively never run generate
        elif decode_ratio >= 1.0:
            N = 1  # Run generate every batch
        else:
            N = max(int(1 / decode_ratio), 1)  # Ensure N is at least 1

        batch_idx = 0

        # Initialize overall metrics
        overall_res = {
            "loss": torch.tensor(0.0, device=self.device),
            "correct": torch.tensor(0.0, device=self.device),
            "total": torch.tensor(0.0, device=self.device),
        }

        # Initialize per-task metrics
        per_task_res = defaultdict(
            lambda: {
                "loss": torch.tensor(0.0, device=self.device),
                "correct": torch.tensor(0.0, device=self.device),
                "total": torch.tensor(0.0, device=self.device),
                "n_sample": 0,
                "predicted_texts": [],
                "gold_texts": [],
            }
        )

        for samples in metric_logger.log_every(dataloader, self.config.run.log_freq, header=header):
            samples = prepare_sample_dist(samples, self.device)

            with torch.autocast(self.device.type, enabled=self.use_amp):
                forward_result = model(samples, verbose=True)

            # Extract batch-level loss and correct counts
            batch_loss = forward_result.get("loss", torch.tensor(0.0, device=self.device))
            batch_correct = forward_result.get("correct", torch.tensor(0.0, device=self.device))
            batch_total = forward_result.get("total", torch.tensor(1.0, device=self.device))

            batch_size = len(samples["id"])

            # Update overall metrics with batch-level values
            overall_res["loss"] += batch_loss.detach()
            overall_res["correct"] += batch_correct.detach()
            overall_res["total"] += batch_total.detach()

            # Decide whether to run generate based on decode_ratio
            if decode and (batch_idx % N == 0):
                prompts = samples.get("prompt", None)
                try:
                    generated_texts = model.generate(samples, self.config.generate, prompts=prompts)
                except Exception as e:
                    print("error in generation", e)
                    generated_texts = [None] * batch_size
            else:
                generated_texts = [None] * batch_size  # Placeholder if not decoding

            # Process per-sample data for per-task metrics and result saving
            for i in range(batch_size):
                task = samples["task"][i]

                # Collect per-task batch-level metrics
                per_task_res[task]["loss"] += batch_loss.detach()
                per_task_res[task]["correct"] += batch_correct.detach()
                per_task_res[task]["total"] += batch_total.detach()
                per_task_res[task]["n_sample"] += 1

                res = {
                    "id": samples["id"][i],
                    "ground_truth": samples["text"][i],  # Gold label from dataloader
                    "task": task,
                    "predicted_text": generated_texts[i],
                }

                if decode and generated_texts[i] is not None:
                    res["prompt"] = samples.get("prompt", [None])[i]

                results_per_task[task].append(res)
                overall_results.append(res)

                # Collect texts for custom metrics
                if generated_texts[i] is not None:
                    per_task_res[task]["predicted_texts"].append(generated_texts[i])
                    per_task_res[task]["gold_texts"].append(samples["text"][i])

            batch_idx += 1  # Increment batch index

        if save_json:
            for task, task_results in results_per_task.items():
                self.save_result(task_results, self.output_dir, f"eval_{split}_{task}_epoch_{epoch}")
            # Optionally save overall results
            self.save_result(overall_results, self.output_dir, f"eval_{split}_epoch_{epoch}")

        # Synchronize metrics across processes if in distributed mode
        if is_dist_avail_and_initialized():
            for key in overall_res:
                dist.all_reduce(overall_res[key])

        overall_ret = {
            "loss": (overall_res["loss"] / batch_idx).item(),
            "agg_metrics": (overall_res["correct"] / overall_res["total"]).item(),
        }

        if is_main_process():
            # Log overall metrics
            wandb.log(
                {
                    f"{split}_loss": overall_ret["loss"],
                    f"{split}_accuracy": overall_ret["agg_metrics"],
                    "epoch": epoch,
                }
            )

        # Compute and log per-task metrics
        for task, res in per_task_res.items():
            if "caption-none" in task:
                continue

            if self.use_distributed:
                print(f"Rank {dist.get_rank()}, task={task}, ")

            print(
                f"loss={res['loss'].shape, res['loss'].dtype}, "
                f"correct={res['correct'].shape, res['correct'].dtype}, "
                f"total={res['total'].shape, res['total'].dtype}, "
                f"n_sample={res['n_sample']}"
            )

            # Synchronize metrics across processes if in distributed mode
            if is_dist_avail_and_initialized():
                dist.all_reduce(res["loss"])
                dist.all_reduce(res["correct"])
                dist.all_reduce(res["total"])
                dist.all_reduce(torch.tensor(res["n_sample"], device=self.device))

            ret = {
                "loss": (res["loss"] / res["n_sample"]).item(),
                "agg_metrics": (res["correct"] / res["total"]).item(),
            }

            if is_main_process():
                # Log per-task metrics
                wandb.log(
                    {
                        f"{split}_{task}_loss": ret["loss"],
                        f"{split}_{task}_accuracy": ret["agg_metrics"],
                        "epoch": epoch,
                    }
                )

                # Get and compute custom metrics for this task
                metrics_list = get_task_metrics(task)
                predicted_texts = res["predicted_texts"]
                gold_texts = res["gold_texts"]
                for metric in metrics_list:
                    if predicted_texts and gold_texts:
                        metric_value = metric.compute_metric(predicted_texts, gold_texts)
                        metric_name = metric.__class__.__name__
                        wandb.log(
                            {
                                f"{split}_{task}_{metric_name}": metric_value,
                                "epoch": epoch,
                            }
                        )
        return overall_ret  # Return overall metrics

    def save_result(self, result, result_dir, filename):
        result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, get_rank()))
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        try:
            json.dump(result, open(result_file, "w"), ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Error saving {result_file}. Error: {e}")
            json.dump(result, open(result_file, "w", encoding="utf-8"), ensure_ascii=False)

        # if is_dist_avail_and_initialized():
        #     dist.barrier()

        if is_main_process():
            logging.info("rank %d starts merging results." % get_rank())
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
                try:
                    res = json.load(open(result_file, "r"))
                except Exception as e:
                    logging.warning(f"Error reading {result_file}. Error: {e}")
                    res = json.load(open(result_file, "r", encoding="utf-8"))
                result += res

            try:
                json.dump(result, open(final_result_file, "w"), ensure_ascii=False)
            except Exception as e:
                logging.warning(f"Error saving {final_result_file}. Error: {e}")
                json.dump(
                    result,
                    open(final_result_file, "w", encoding="utf-8"),
                    ensure_ascii=False,
                )

            print("result file saved to %s" % final_result_file)

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if self.evaluate_only:
                break

            # training phase
            logging.info("Training Phase")
            train_stats = self.train_epoch(cur_epoch)
            self.log_stats(train_stats, split_name="train")

            # validating phase
            logging.info("Validating Phase")
            valid_log = self.valid_epoch(
                cur_epoch,
                "valid",
                decode=self.config.run.custom_metrics,
                save_json=False,
                decode_ratio=self.config.run.decode_ratio,
            )
            if valid_log is not None:
                if is_main_process():
                    agg_metrics = valid_log["agg_metrics"]
                    if agg_metrics > best_agg_metric:
                        best_agg_metric = agg_metrics
                        best_epoch = cur_epoch
                        self.save_checkpoint(cur_epoch, is_best=True)

                    valid_log.update({"best_epoch": best_epoch})
                    self.log_stats(valid_log, split_name="valid")
            self.save_checkpoint(cur_epoch, is_best=False)

            # if self.use_distributed:
            #     dist.barrier()

        # testing phase
        if self.evaluate_only:
            self.valid_epoch("best", "test", decode=True, save_json=True)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.model_dump_json(), indent=4) + "\n")

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": dict(self.config),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)
