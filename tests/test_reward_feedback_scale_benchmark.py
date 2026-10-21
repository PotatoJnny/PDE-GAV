from scripts.benchmark_ns_reward_feedback_scale import _summarize


def _row(size: int, spearman: float, recall: float) -> dict[str, float | int]:
    return {
        "train_size": size,
        "clean_spearman": spearman + 0.05,
        "guidance_path_spearman": spearman,
        "guidance_path_hard_quartile_recall": recall,
        "guidance_path_top_quartile_lift": 1.5,
        "training_seconds": 2.0,
    }


def test_plateau_rule_selects_the_smallest_near_largest_model() -> None:
    sizes = (64, 128, 256, 512, 1024)
    quality = {
        64: (0.40, 0.60),
        128: (0.50, 0.67),
        256: (0.61, 0.73),
        512: (0.69, 0.78),
        1024: (0.70, 0.80),
    }
    rows = [
        _row(size, quality[size][0] + offset, quality[size][1] + offset)
        for offset in (-0.005, 0.0, 0.005)
        for size in sizes
    ]

    summary = _summarize(
        rows,
        train_sizes=sizes,
        spearman_tolerance=0.02,
        recall_tolerance=0.03,
    )

    assert summary["recommended_feedback_case_count"] == 512
    assert not summary["by_train_size"]["256"]["within_largest_size_plateau"]
    assert summary["by_train_size"]["512"]["within_largest_size_plateau"]
    assert summary["by_train_size"]["1024"]["within_largest_size_plateau"]
