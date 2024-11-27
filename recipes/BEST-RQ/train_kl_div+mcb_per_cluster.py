#!/usr/bin/env python3
"""Recipe for pretraining BestRQ
See config file for model definition.
See the readme of the recipe for advice on the pretraining that may appear
a bit challenging depending on your available resources.

To run this recipe call python train_kl_div+mcb_per_cluster.py best_rq.yaml --find_unused_parameters --max_grad_norm 0.0

Authors
    * Ryan Whetten 2023
    * Ilja Baumann 2024
"""

import logging
import sys
import os
import time
from functools import partial

import speechbrain as sb
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from hyperpyyaml import load_hyperpyyaml

from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.sampler import DynamicBatchSampler
from mask import brq_with_label_mask_collate_fn

logger = logging.getLogger(__name__)


class BestRQBrain(sb.core.Brain):

    def compute_forward(self, batch, stage):
        """Computes forward pass through BestRQ model and returns encoded and
        target embeddings as well as other metrics of interest.
        """
        # get batch and mask
        wavs, wav_lens, mask, _ = batch
        wavs, wav_lens, mask = (
            wavs.to(self.device),
            wav_lens.to(self.device),
            mask.to(self.device)
        )
        ############### START ##############
        ### get fbanks and normalize
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        B, T, C = feats.shape
        divis_by = self.hparams.pad_to_divisible_by

        #### pad features
        dim_to_pad = 1  # Pad along the second dimension (i.e. time)

        # Calculate the amount of padding needed to make the tensor divisible by 4
        current_dim_size = feats.shape[dim_to_pad]
        padding_needed = (4 - (current_dim_size % 4)) % 4  # Ensure positive padding

        # Define the padding
        padding = [0, 0, 0, 0, 0, 0]  # Initialize padding for all dimensions
        padding[dim_to_pad * 2] = padding_needed  # Set padding for the chosen dimension

        # add in padding to features and mask
        feats = torch.nn.functional.pad(feats, padding)

        # get targets from quantizer
        if self.hparams.loss_use_kl_div:
            targets, distances = self.modules.Quantizer(feats.view(B, feats.shape[1] // divis_by, -1))
        else:
            targets = self.modules.Quantizer(feats.view(B, feats.shape[1] // divis_by, -1))

        ### augment data
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                feats = self.hparams.augmentation(feats)

        # generate random noise
        noise = torch.normal(
            mean=self.hparams.noise_mean,
            std=self.hparams.noise_std,
            size=(B, mask.shape[0], C),
            device=self.device
        )
        # replace with random noise
        feats[:, mask, :] = noise

        #### convolutions
        src = self.modules.CNN(feats)

        ##### transformer
        enc_out = self.modules.wrapper(src, wav_lens)  # only use encoder

        ##### linear layers
        linear_modules = self.modules.Linears.module if isinstance(self.modules.Linears,
                                                                   torch.nn.parallel.DistributedDataParallel) else self.modules.Linears
        logits_list = [linear(enc_out) for linear in linear_modules]

        mask_idx = mask[::divis_by] // divis_by


        if self.hparams.loss_use_kl_div:
            distances_list = []
        for i, logits in enumerate(logits_list):
            logits_list[i] = logits[:, mask_idx, :]
            if self.hparams.loss_use_kl_div:
                distances_list.append(distances[i][:, mask_idx, :])
        targets = targets[:, mask_idx]
        B, T, C = logits_list[0].shape

        result = tuple(l.view(B * T, C) for l in logits_list) + (targets.reshape(B * T, self.hparams.cb_num),)
        if self.hparams.loss_use_kl_div:
            return result, tuple(d.view(B * T, C) for d in distances_list)

        return result

    def compute_codebook_perplexity(self, outputs):
        if self.hparams.loss_use_kl_div:
            predictions, distances = outputs
        else:
            predictions = outputs
        codebooks = self.hparams.cb_num
        targets = torch.split(predictions[codebooks], 1, dim=1)
        perplexities = []
        for i in range(codebooks):
            unique_codes = torch.unique(targets[i].squeeze())
            counts = torch.tensor([(targets[i].squeeze() == code).sum().item() for code in unique_codes], dtype=torch.float32)
            probabilities = counts / counts.sum()
            perplexity = torch.exp(-torch.sum(probabilities * torch.log(probabilities)))
            perplexities.append(perplexity.item())
        return sum(perplexities) / len(perplexities)

    def compute_objectives(self, outputs, batch, stage):
        if self.hparams.loss_use_kl_div:
            predictions, distances = outputs
        else:
            predictions = outputs
        num_codebooks = self.hparams.cb_num
        pred_list = predictions[:num_codebooks]
        targets = torch.split(predictions[num_codebooks], 1, dim=1)

        # Compute accuracy if not in training stage
        if stage != sb.Stage.TRAIN and sb.utils.distributed.if_main_process():
            accuracies = []
            for i, pred in enumerate(pred_list):
                predicted_classes = torch.argmax(pred, dim=-1)
                correct_predictions = (predicted_classes == targets[i].squeeze())
                accuracy = correct_predictions.sum().item() / len(correct_predictions)
                accuracies.append(accuracy)
            avg_accuracy = sum(accuracies) / num_codebooks
            self.acc_metric.append(avg_accuracy)

        _, _, _, batch_clusters = batch
        device = predictions[0].device

        T = int(len(pred_list[0]) / len(batch_clusters))
        B = batch_clusters.size(0)

        primary_weight = self.hparams.cb_primary_weight
        secondary_weight = self.hparams.cb_secondary_weight

        weights = torch.full((num_codebooks, B * T), secondary_weight, device=device)
        for c in range(num_codebooks):
            for b in range(B):
                if batch_clusters[b].item() == c:
                    weights[c, b * T:(b + 1) * T] = primary_weight

        losses = []
        if self.hparams.loss_use_ce:
            ce_losses = []
            for c in range(num_codebooks):
                loss = F.cross_entropy(pred_list[c], targets[c].squeeze(), reduction='none')

                weighted_loss = loss * weights[c]
                weighted_loss = weighted_loss.sum() / (B * T)

                ce_losses.append(weighted_loss)
            ce_loss = (sum(ce_losses) / num_codebooks) * self.hparams.loss_ce_weight
            losses.append(ce_loss)

        if self.hparams.loss_use_kl_div:
            kl_losses = []
            for c in range(num_codebooks):
                cb_pred = pred_list[c]
                cb_dist = distances[c]

                loss = F.kl_div(
                    F.log_softmax(cb_pred, dim=-1),
                    F.softmax(cb_dist, dim=-1),
                    reduction="none",
                )

                loss = torch.clamp(loss, min=0)
                loss = torch.sum(loss, dim=-1)

                weighted_loss = loss * weights[c]
                weighted_loss = weighted_loss.sum() / (B * T)
                kl_losses.append(weighted_loss)

            kl_loss = (sum(kl_losses) / num_codebooks) * self.hparams.loss_kl_div_weight
            losses.append(kl_loss)

        loss = sum(losses) / len(losses)
        return loss

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)

                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)

                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                        device_type=torch.device(self.device).type,
                        dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_batch_end(self, batch, outputs, loss, should_ste):
        """ Called after fit_batch(), updates learning rate and does per-step logging. """

        # Perform step-wise logging
        if (
                hasattr(self.hparams, "log_interval")
                and self.optimizer_step % self.hparams.log_interval == 0
        ):

            # Create a dictionary and fill it with everything we
            # want to log such as contrastive loss, diversity loss,
            # learning rate etc.
            log_dct = {}

            current_lr = self.optimizer.param_groups[0]["lr"]
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = current_lr
            log_dct["avg_loss"] = self.avg_train_loss

            # Compute codebook perplexity
            codebook_perplexity = self.compute_codebook_perplexity(outputs)
            log_dct["codebook_perplexity"] = codebook_perplexity

            if hasattr(self, "time_last_log"):
                run_time_since_last_log = time.time() - self.time_last_log
                log_dct["run_time"] = run_time_since_last_log
            self.time_last_log = time.time()

            if sb.utils.distributed.if_main_process():
                self.hparams.train_steps_logger.log_stats(stats_meta=log_dct, )

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = []

    def on_stage_end(self, stage, stage_loss, epoch=None):

        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:
            if self.acc_metric:
                stage_stats["accuracy"] = sum(self.acc_metric) / len(
                    self.acc_metric
                )

            self.hparams.train_stage_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "steps": self.optimizer_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                end_of_epoch=True,
                num_to_keep=8,
                meta={"valid_loss": stage_loss},
            )


def dataio_prepare(hparams):
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    # We remove longer and shorter files from the train.
    train_data = train_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_shorter_than"]},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data]

    def get_output_lengths(input_lengths):
        """ Function to get the output length of the feature extractor this is
            necessary to compute the masks of BestRQ.
        """
        sr = hparams["sample_rate"]
        hop_length = hparams["hop_length"]

        return (input_lengths // (sr * hop_length / 1000) + 1).to(torch.long)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)

            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    @sb.utils.data_pipeline.takes("cluster")
    @sb.utils.data_pipeline.provides("label")
    def label_pipeline(cluster):
        label = label_encoder.encode_label_torch(cluster)
        yield label

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets[0]],  # train
        output_key="cluster",
        sequence_input=False,
    )

    label_encoder.expect_len(hparams["cb_num"])

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "label"])


    # We create the DynamicBatch Sampler
    train_sampler = DynamicBatchSampler(
        train_data,
        hparams["seconds_per_batch"],
        num_buckets=hparams["train_num_buckets"],
        length_func=lambda x: x["duration"],
        batch_ordering="random",
        shuffle=True,
    )

    # We define the custom collation function that is necessary for best-rq to
    # generate masks.
    brq_mask_collate_fn_partial = partial(
        brq_with_label_mask_collate_fn,
        get_out_len_fn=get_output_lengths,
        mask_prob=hparams["mask_prob"],
        mask_length=hparams["mask_length"],
        n_mels=hparams["n_mels"],
    )

    train_loader_kwargs = {
        "batch_sampler": train_sampler,
        "collate_fn": brq_mask_collate_fn_partial,
        "num_workers": hparams["train_dataloader_options"]["num_workers"],
        "pin_memory": True,
    }

    valid_loader = SaveableDataLoader(
        valid_data,
        collate_fn=brq_mask_collate_fn_partial,
        num_workers=hparams["test_dataloader_options"]["num_workers"],
        batch_size=hparams["test_dataloader_options"]["batch_size"],
        pin_memory=True,
    )

    return train_data, valid_loader, train_loader_kwargs


def main():
    logger.setLevel(logging.INFO)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams.update(run_opts)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from librispeech_prepare import prepare_librispeech

    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # Part that matters starts here.
    train_dataset, valid_loader, train_loader_kwargs = dataio_prepare(hparams)

    brain = BestRQBrain(
        modules=hparams["modules"],
        opt_class=hparams["optimizer"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    # with torch.autograd.detect_anomaly():
    brain.fit(
        brain.hparams.epoch_counter,
        train_dataset,
        valid_loader,
        train_loader_kwargs=train_loader_kwargs,
        progressbar=True,
    )


if __name__ == "__main__":
    main()
