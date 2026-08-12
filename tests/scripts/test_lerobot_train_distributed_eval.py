import pytest

from lerobot.scripts.lerobot_train import _merge_distributed_eval_infos


def _rank_info(rank, task_id, successes):
    per_episode = [
        {
            "episode_ix": index,
            "success": success,
            "sum_reward": float(success),
            "max_reward": float(success),
            "mask_type": "random_mask",
            "mask_type_slot": task_id,
        }
        for index, success in enumerate(successes)
    ]
    return {
        "rank": rank,
        "task_ids": [task_id],
        "eval_info": {
            "per_task": [
                {
                    "task_group": "libero_10",
                    "task_id": task_id,
                    "metrics": {
                        "sum_rewards": [episode["sum_reward"] for episode in per_episode],
                        "max_rewards": [episode["max_reward"] for episode in per_episode],
                        "successes": successes,
                        "video_paths": [],
                    },
                }
            ],
            "per_episode": per_episode,
            "overall": {
                "n_episodes": len(successes),
                "pc_success": sum(successes) / len(successes) * 100,
                "avg_sum_reward": sum(successes) / len(successes),
                "eval_s": 10.0 + rank,
            },
        },
    }


def test_merge_distributed_mam_eval_uses_episode_weighting_and_keeps_mask_metrics():
    merged = _merge_distributed_eval_infos(
        [
            _rank_info(rank=0, task_id=0, successes=[True, False]),
            _rank_info(rank=1, task_id=1, successes=[True]),
        ]
    )

    assert merged["overall"]["n_episodes"] == 3
    assert merged["overall"]["pc_success"] == pytest.approx(200 / 3)
    assert merged["overall"]["avg_sum_reward"] == pytest.approx(2 / 3)
    assert merged["overall"]["eval_s"] == 11.0
    assert merged["per_mask_type_success"] == {"random_mask": pytest.approx(200 / 3)}
    assert merged["per_mask_slot_success"] == {"0": 50.0, "1": 100.0}
    assert [episode["episode_ix"] for episode in merged["per_episode"]] == [0, 1, 2]
    assert [episode["rank"] for episode in merged["per_episode"]] == [0, 0, 1]
