import math


LR_SCHEDULERS = ("constant", "cosine")


def get_learning_rate(
    current_step,
    total_steps,
    peak_lr,
    min_lr,
    warmup_steps,
    scheduler_name,
):
    """Return the LR for the current zero-based optimizer update."""
    if scheduler_name not in LR_SCHEDULERS:
        raise ValueError(
            f"不支持的 lr_scheduler: {scheduler_name}; "
            f"可选值为 {', '.join(LR_SCHEDULERS)}"
        )
    if total_steps <= 0:
        raise ValueError("total_steps 必须大于 0")
    if current_step < 0:
        raise ValueError("current_step 不能小于 0")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps 必须满足 0 <= warmup_steps < total_steps")

    if scheduler_name == "constant":
        return peak_lr

    if warmup_steps > 0 and current_step < warmup_steps:
        return peak_lr * current_step / warmup_steps

    decay_steps = total_steps - warmup_steps
    decay_progress = (current_step - warmup_steps) / decay_steps
    decay_progress = min(max(decay_progress, 0.0), 1.0)

    return min_lr + (peak_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )


def build_lr_schedule_config(args, steps_per_epoch):
    scheduler_name = str(getattr(args, "lr_scheduler", "cosine"))
    warmup_ratio = float(getattr(args, "warmup_ratio", 0.1))
    min_lr = float(getattr(args, "min_lr", 1e-6))
    peak_lr = float(args.lr)

    if scheduler_name not in LR_SCHEDULERS:
        raise ValueError(
            f"不支持的 lr_scheduler: {scheduler_name}; "
            f"可选值为 {', '.join(LR_SCHEDULERS)}"
        )
    if steps_per_epoch <= 0:
        raise ValueError("训练 DataLoader 为空，无法计算学习率计划")
    if args.epochs <= 0:
        raise ValueError("epochs 必须大于 0")
    if peak_lr <= 0:
        raise ValueError("lr 必须大于 0")
    if scheduler_name == "cosine" and (min_lr < 0 or min_lr > peak_lr):
        raise ValueError("min_lr 必须满足 0 <= min_lr <= lr")
    if warmup_ratio < 0 or warmup_ratio >= 1:
        raise ValueError("warmup_ratio 必须满足 0 <= warmup_ratio < 1")

    total_steps = args.epochs * steps_per_epoch
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)

    warmup_steps = (
        int(total_steps * warmup_ratio)
        if scheduler_name == "cosine"
        else 0
    )

    return {
        "scheduler_name": scheduler_name,
        "peak_lr": peak_lr,
        "min_lr": min_lr,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }


def set_optimizer_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group.get("lr_scale", 1.0)
