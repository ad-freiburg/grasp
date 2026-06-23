import argparse
import math
import os
import random
from logging import Logger
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, get_peft_model
from pydantic import BaseModel
from torch.utils.data import ConcatDataset, Dataset, Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import load_json
from universal_ml_utils.logging import get_logger

from grasp.baselines.grisp.data import (
    IGNORE_INDEX,
    GRISPCollator,
    GRISPMaterializedSelectionDataset,
    GRISPMaterializedSkeletonDataset,
    GRISPSelectionDataset,
    GRISPSkeletonDataset,
    load_samples,
)
from grasp.baselines.grisp.utils import (
    SeededRandomSampler,
    find_best_checkpoint,
    find_latest_checkpoint,
    set_chat_template,
)
from grasp.configs import KgConfig
from grasp.manager import load_kg_manager


class Lora(BaseModel):
    r: int = 32
    lora_alpha: int = 32
    target_modules: list[str] | str = "all-linear"
    save_modules: list[str] | None = None
    dropout: float = 0.05


class GRISPTrainConfig(BaseModel):
    # model
    model: str
    overwrite_chat_template: bool = False
    do_compile: bool = False
    lora: Lora | None = None

    # data
    type: str
    train_files: list[str]
    val: list[str] | float
    materialized: bool = False
    max_length: int = 8192
    mask_inputs: bool = True
    num_workers: int = 4
    knowledge_graph: KgConfig | None = None

    # data augmentation
    skeleton_p: float = 0.2
    drop_infos_p: float = 0.05
    drop_target_p: float = 0.1
    shuffle_alts_p: float = 0.1

    # training hyperparameters
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    batch_size: int = 8
    num_epochs: int | float = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    seed: int = 22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRISP model")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the training artifacts",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def load_from_run_directory(
    directory: str,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase, Lora | None]:
    # warm-start from a previous GRISP run dir, mirroring run.py: pick its best
    # checkpoint and load base+adapter (if it was LoRA) or the full model.
    checkpoint = find_best_checkpoint(directory)
    assert checkpoint is not None, f"No checkpoint found in run directory {directory}"

    train_cfg = GRISPTrainConfig(**load_config(os.path.join(directory, "config.yaml")))
    if train_cfg.lora is not None:
        model = AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            dtype="auto",
            is_trainable=True,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            dtype="auto",
            attn_implementation="eager",
        )

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)  # type: ignore
    return model, tokenizer, train_cfg.lora


def load_model_and_tokenizer(
    config: GRISPTrainConfig,
) -> tuple[PreTrainedModel | PeftModel, PreTrainedTokenizerBase]:
    logger = get_logger("GRISP TRAIN")
    is_run_dir = os.path.isdir(config.model) and os.path.exists(
        os.path.join(config.model, "config.yaml")
    )
    loaded_lora: Lora | None = None
    if is_run_dir:
        model, tokenizer, loaded_lora = load_from_run_directory(config.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            dtype="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model)

    if config.overwrite_chat_template:
        tokenizer = set_chat_template(tokenizer)

    if isinstance(model, PeftModel):
        # warm-started from a previous LoRA run: continue training the loaded
        # adapter as is. Its structure is fixed, so adopt the loaded run's lora
        # config (comparing against the saved config, not the adapter, since e.g.
        # "all-linear" gets expanded into a concrete module list on save).
        # Adopting it also keeps the saved run marked as LoRA, which run.py keys
        # its loading on.
        if config.lora is not None and config.lora != loaded_lora:
            logger.warning(
                "Specified `lora` differs from the loaded adapter; continuing with "
                "the loaded run's lora config instead.\n"
                f"  loaded:    {loaded_lora}\n  specified: {config.lora}"
            )
        config.lora = loaded_lora
    elif config.lora is not None:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.dropout,
            modules_to_save=config.lora.save_modules,
        )
        model = get_peft_model(model, peft_config)
        assert isinstance(model, PeftModel)

    return model, tokenizer


def load_datasets(
    cfg: GRISPTrainConfig,
    tokenizer: PreTrainedTokenizerBase,
    logger: Logger,
) -> tuple[Dataset, Dataset]:
    if cfg.type == "both":
        cfg.type = "skeleton"
        train_skel, val_skel = load_datasets(
            cfg,
            tokenizer,
            logger,
        )
        cfg.type = "selection"
        train_sel, val_sel = load_datasets(
            cfg,
            tokenizer,
            logger,
        )
        cfg.type = "both"
        train_data = ConcatDataset([train_skel, train_sel])
        val_data = ConcatDataset([val_skel, val_sel])
        return train_data, val_data

    samples = load_samples(cfg.train_files, cfg.materialized)
    dataset_kwargs = {
        "samples": samples,
        "tokenizer": tokenizer,
        "mask_inputs": cfg.mask_inputs,
        "log_level": logger.level,
    }
    if cfg.type == "skeleton":
        if cfg.materialized:
            dataset_cls = GRISPMaterializedSkeletonDataset
        else:
            dataset_cls = GRISPSkeletonDataset
            dataset_kwargs["p"] = cfg.skeleton_p

    elif cfg.type == "selection":
        if cfg.materialized:
            dataset_cls = GRISPMaterializedSelectionDataset
        else:
            dataset_cls = GRISPSelectionDataset
            assert cfg.knowledge_graph is not None, (
                "KG config must be provided for selection dataset"
            )
            manager = load_kg_manager(cfg.knowledge_graph)
            dataset_kwargs["manager"] = manager
            dataset_kwargs["skeleton_p"] = cfg.skeleton_p
            dataset_kwargs["drop_infos_p"] = cfg.drop_infos_p
            dataset_kwargs["drop_target_p"] = cfg.drop_target_p
            dataset_kwargs["shuffle_alts_p"] = cfg.shuffle_alts_p
            logger.warning("Setting num workers to 0 for online selection training")
            cfg.num_workers = 0
    else:
        raise ValueError(f"Unknown train type: {cfg.type}")

    train_data = dataset_cls(**dataset_kwargs)

    if isinstance(cfg.val, list):
        dataset_kwargs["samples"] = load_samples(cfg.val, cfg.materialized)
        if not cfg.materialized:
            dataset_kwargs["is_val"] = True

        val_data = dataset_cls(**dataset_kwargs)
        return train_data, val_data

    assert cfg.val > 0 and cfg.val < 1.0, "Val split size must be a float in (0, 1)"
    indices = list(range(len(train_data)))
    random.seed(cfg.seed)
    random.shuffle(indices)
    num_val_samples = round(len(train_data) * cfg.val)
    num_val_samples = max(1, min(num_val_samples, len(train_data) - 1))
    val_indices = indices[:num_val_samples]
    train_indices = indices[num_val_samples:]
    logger.info(
        f"Splitting data into {len(train_indices):,} train "
        f"and {len(val_indices):,} val samples"
    )

    val_samples = [train_data.samples[i] for i in val_indices]
    train_samples = [train_data.samples[i] for i in train_indices]

    dataset_kwargs["samples"] = train_samples
    train_data = dataset_cls(**dataset_kwargs)

    dataset_kwargs["samples"] = val_samples
    if not cfg.materialized:
        dataset_kwargs["is_val"] = True

    val_data = dataset_cls(**dataset_kwargs)
    return train_data, val_data


def advance_dataset(
    dataset: Dataset,
    seed: int,
    epochs_trained: int,
    batch_size: int,
    batches_in_current_epoch: int,
) -> None:
    n = len(dataset)  # type: ignore
    sampler = SeededRandomSampler(n, seed)

    # past epochs
    for _ in range(epochs_trained):
        for idx in sampler:
            # access to trigger counter updates
            _ = dataset[idx]

    num_seen = min(batches_in_current_epoch * batch_size, n)

    # partial epoch
    i = 0
    for idx in sampler:
        if i >= num_seen:
            break
        # access to trigger counter updates
        _ = dataset[idx]
        i += 1


class GRISPTrainer(Trainer):
    def __init__(self, *args, epochs_trained: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epochs_trained = epochs_trained
        # compute_loss normalizes per micro-batch, so let Trainer re-apply its
        # /gradient_accumulation_steps division (else loss/grads scale with it).
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:  # type: ignore
        if dataset is None:
            dataset = self.train_dataset  # type: ignore
        return SeededRandomSampler(
            len(dataset),  # type: ignore
            seed=self.args.seed,
            epoch=self.epochs_trained,
        )

    def compute_loss(  # type: ignore
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any] | torch.Tensor:
        option_ids = inputs.pop("option_token_ids")
        option_mask = inputs.pop("option_mask")
        target_idx = inputs.pop("target_idx")
        answer_pos = inputs.pop("answer_pos")
        is_select = inputs.pop("is_selection")
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits  # (B, T, V)
        T = logits.size(1)

        total_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        total_n = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if (~is_select).any():
            skel_logits = logits[~is_select]
            skel_labels = labels[~is_select]
            shift_logits = skel_logits[:, :-1].contiguous()
            shift_labels = skel_labels[:, 1:].contiguous()
            ntp = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            n_ntp = (shift_labels != IGNORE_INDEX).sum().to(logits.dtype)
            total_loss = total_loss + ntp
            total_n = total_n + n_ntp

        # Drop selection rows whose answer position fell beyond the (possibly
        # truncated) sequence; indexing them would go out of bounds.
        sel_valid = is_select & (answer_pos >= 1) & (answer_pos < T)
        if sel_valid.any():
            sel_rows = sel_valid.nonzero(as_tuple=True)[0]
            # logit at position t-1 predicts token at position t
            pred_pos = answer_pos[sel_rows] - 1
            ans_logits = logits[sel_rows, pred_pos]  # (M, V)
            opt_logits = ans_logits.gather(1, option_ids[sel_rows])  # (M, K)
            opt_logits = opt_logits.masked_fill(~option_mask[sel_rows], float("-inf"))
            sel = F.cross_entropy(opt_logits, target_idx[sel_rows], reduction="sum")
            n_sel = torch.tensor(
                len(sel_rows), device=logits.device, dtype=logits.dtype
            )
            total_loss = total_loss + sel
            total_n = total_n + n_sel

        loss = total_loss / total_n.clamp(min=1)
        return (loss, outputs) if return_outputs else loss


def main(args: argparse.Namespace) -> None:
    logger = get_logger("GRISP TRAIN", args.log_level)

    config = GRISPTrainConfig(**load_config(args.config))

    model, tokenizer = load_model_and_tokenizer(config)

    if config.gradient_checkpointing:
        # get rid of incompatibility warning
        model.config.use_cache = False  # type: ignore

    logger.info(f"Using model:\n{model}")
    total = model.num_parameters()  # type: ignore
    logger.info(f"Total parameters: {total / 1e9:.1f}B")
    trainable = model.num_parameters(only_trainable=True)  # type: ignore
    logger.info(
        f"Trainable parameters: {trainable / 1e6:.2f}M ({trainable / total:.2%})"
    )

    if config.materialized and config.num_workers > 0:
        logger.warning(
            "Materialized datasets cannot be used with multiple workers. "
            "Setting num_workers to 0."
        )
        config.num_workers = 0

    train_data, val_data = load_datasets(config, tokenizer, logger)
    collator = GRISPCollator(
        tokenizer.pad_token_id,  # type: ignore
        config.max_length,
        args.log_level,
    )

    run_name = os.path.basename(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = find_latest_checkpoint(args.output_dir)

    logger.info(f"Train dataset size: {len(train_data):,} samples")  # type: ignore
    logger.info(f"Validation dataset size: {len(val_data):,} samples")  # type: ignore

    # save config
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config.model_dump(), f)

    # config.batch_size is the global (effective) batch size across data-parallel
    # replicas and gradient-accumulation micro-batches. Derive the per-device
    # micro-batch size HF expects, requiring clean divisibility.
    world_size = max(1, torch.cuda.device_count())
    accum = config.gradient_accumulation_steps
    denom = world_size * accum
    if config.batch_size % denom != 0:
        raise ValueError(
            f"Global batch_size ({config.batch_size}) must be divisible by "
            f"world_size ({world_size}) * gradient_accumulation_steps ({accum}) = {denom}"
        )
    per_device_batch_size = config.batch_size // denom
    # samples per forward across all replicas (one accumulation micro-batch)
    dataloader_batch_size = per_device_batch_size * world_size
    logger.info(
        f"Global batch size {config.batch_size} = per_device {per_device_batch_size} "
        f"x world_size {world_size} x grad_accum {accum}"
    )

    batches_per_epoch = math.ceil(len(train_data) / dataloader_batch_size)  # type: ignore
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    logging_steps = max(1, steps_per_epoch // 100)  # log 100 times per epoch

    # eval once per epoch, but at least 10 times during training
    total_steps = int(steps_per_epoch * config.num_epochs)
    eval_steps = max(1, min(steps_per_epoch, total_steps // 10))

    report_to = None
    if os.environ.get("WANDB_PROJECT"):
        os.environ["WANDB_NAME"] = run_name
        report_to = "wandb"

    epochs_trained = 0
    if checkpoint is not None:
        logger.info(f"Resuming training from checkpoint {checkpoint}")
        trainer_state = load_json(os.path.join(checkpoint, "trainer_state.json"))
        global_step = trainer_state["global_step"]
        epochs_trained = global_step // steps_per_epoch
        steps_in_current_epoch = global_step % steps_per_epoch
        batches_in_current_epoch = (
            steps_in_current_epoch * config.gradient_accumulation_steps
        )

        if config.materialized:
            # materialized datasets have counters that track seen samples,
            # so we need to restore the correct counter state
            advance_dataset(
                train_data,
                seed=config.seed,
                epochs_trained=epochs_trained,
                batch_size=dataloader_batch_size,
                batches_in_current_epoch=batches_in_current_epoch,
            )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        # only eval_loss is used; skip keeping logits to cut eval peak memory
        prediction_loss_only=True,
        save_strategy="steps",
        save_total_limit=1,
        save_steps=eval_steps,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        warmup_steps=int(total_steps * config.warmup_ratio),
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_epochs,
        seed=config.seed,
        bf16=True,
        report_to=report_to,
        run_name=run_name,
        metric_for_best_model="eval_loss",
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        torch_compile=config.do_compile,
        dataloader_num_workers=config.num_workers,
        dataloader_prefetch_factor=4 if config.num_workers > 0 else None,
        # keep selection-specific keys (option_token_ids, answer_pos, etc.)
        # that compute_loss uses for restricted CE
        remove_unused_columns=False,
    )

    trainer = GRISPTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator,
        callbacks=[
            EarlyStoppingCallback(max(10, round(config.num_epochs / 10))),
        ],
        epochs_trained=epochs_trained,
    )

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main(parse_args())
