# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib.util
import json
from pathlib import Path

from nemo_gym.rollout_collection import (
    RolloutCollectionConfig,
    collect_sample_metrics,
    format_collect_timing,
    percentile,
    step_timing_breakdown,
    summarize_collect_timing,
)


def _load_collect_timing_script():
    path = Path(__file__).resolve().parents[3] / "scripts" / "collect_timing.py"
    spec = importlib.util.spec_from_file_location("collect_timing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# TODO: Eventually we want to add more tests to ensure that the rollout collection flow does not break
class TestRolloutCollection:
    def test_sanity(self) -> None:
        RolloutCollectionConfig(
            agent_name="",
            input_jsonl_fpath="",
            output_jsonl_fpath="",
        )

    def test_percentile_single_value_is_that_value(self) -> None:
        assert percentile([35.77], 0.95) == 35.77
        assert percentile([35.77], 0.50) == 35.77

    def test_percentile_linear_interpolation(self) -> None:
        # 20 evenly spaced 1..20; p95 at index 0.95 * 19 = 18.05 → 19.05
        values = [float(i) for i in range(1, 21)]
        assert abs(percentile(values, 0.95) - 19.05) < 1e-9

    def test_summarize_p95_and_throughput(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0]
        # concurrent: wall shorter than sum
        timing = format_collect_timing(summarize_collect_timing(latencies, wall_s=40.0))
        assert timing["n"] == 4
        assert timing["latency_mean_s"] == 25.0
        assert timing["latency_p50_s"] == 25.0
        assert timing["latency_p95_s"] == 38.5
        assert timing["latency_max_s"] == 40.0
        assert timing["wall_s"] == 40.0
        assert timing["samples_per_min"] == 6.0

    def test_collect_timing_jsonl_p95_and_throughput(self, tmp_path: Path) -> None:
        script = _load_collect_timing_script()
        path = tmp_path / "out.jsonl"
        job_t0 = 1_000.0
        rows = [
            {"collect_latency_s": 10.0, "collect_job_t0": job_t0, "collect_finished_at": 1_010.0},
            {"collect_latency_s": 20.0, "collect_job_t0": job_t0, "collect_finished_at": 1_020.0},
            {"collect_latency_s": 30.0, "collect_job_t0": job_t0, "collect_finished_at": 1_030.0},
            {"collect_latency_s": 40.0, "collect_job_t0": job_t0, "collect_finished_at": 1_040.0},
        ]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report = script.timing_from_jsonl(str(path))[0]
        assert report["n"] == 4
        assert report["latency_p95_s"] == 38.5
        assert report["wall_s"] == 40.0
        assert report["samples_per_min"] == 6.0

    def test_collect_timing_jsonl_without_latency_fields(self, tmp_path: Path) -> None:
        script = _load_collect_timing_script()
        path = tmp_path / "old.jsonl"
        path.write_text(json.dumps({"reward": 0.5}) + "\n")
        report = script.timing_from_jsonl(str(path))[0]
    def test_collect_sample_metrics_includes_gold_count(self) -> None:
        sample = collect_sample_metrics(
            {
                "id": "row-1",
                "reward": 0.767,
                "factuality_f1_score": 0.667,
                "num_errors": 3,
                "predicted_severity": 5.0,
                "severity_reward": 1.0,
                "hallucination_severity": 5.0,
                "expected_errors": ["a", "b", "c"],
                "collect_latency_s": 35.77,
            },
            done=4,
            n=156,
        )
        assert sample["done"] == 4
        assert sample["n"] == 156
        assert sample["n_expected_errors"] == 3
        assert sample["gold_empty"] is False
        assert sample["factuality_f1_score"] == 0.667

    def test_collect_sample_metrics_passes_through_timings(self) -> None:
        sample = collect_sample_metrics(
            {
                "reward": 1.0,
                "timings": {"policy_loop_s": 12.3, "verify_s": 4.1, "run_s": 16.4},
            },
            done=1,
            n=1,
        )
        assert sample["timing_breakdown"]["total_s"] == 16.4
        assert "timings" not in sample

    def test_collect_sample_metrics_uses_input_gold(self) -> None:
        sample = collect_sample_metrics(
            {"reward": 1.0, "factuality_f1_score": 1.0},
            done=1,
            n=10,
            gold=[],
            sample_id="empty-1",
        )
        assert sample["id"] == "empty-1"
        assert sample["gold_empty"] is True
        assert sample["n_expected_errors"] == 0

    def test_step_timing_breakdown_matches_table(self) -> None:
        breakdown = step_timing_breakdown(
            {
                "judge_count_s": 42.9378,
                "judge_yes_no_s": 24.3516,
                "policy_generate_s": 9.1681,
                "search_wiki_http_s": 0.4957,
                "seed_session_s": 0.0136,
                "run_s": 77.0186,
                "policy_steps": [{}] * 7,
                "searches": [{}] * 6,
                "judge_count_attempts": 3,
                "search_cached": 4,
            }
        )
        assert breakdown["count_judge_s"] == 42.9378
        assert breakdown["yes_no_matcher_s"] == 24.3516
        assert breakdown["policy_generate_s"] == 9.1681
        assert breakdown["search_s"] == 0.4957
        assert breakdown["seed_s"] == 0.0136
        assert breakdown["n_policy_steps"] == 7
        assert breakdown["judges_wall_s"] == 42.9378
        assert breakdown["steps"][0]["label"] == "Count (line-count)"
        assert breakdown["steps"][2]["label"] == "Policy generate (7 steps)"
        assert abs(breakdown["count_judge_share"] - 42.9378 / 77.0186) < 1e-6

    def test_collect_sample_metrics_flattens_timing_breakdown(self) -> None:
        sample = collect_sample_metrics(
            {
                "id": "chris-rock",
                "reward": 1.0,
                "timings": {
                    "judge_count_s": 42.9378,
                    "judge_yes_no_s": 24.3516,
                    "policy_generate_s": 9.1681,
                    "search_wiki_http_s": 0.4957,
                    "seed_session_s": 0.0136,
                    "run_s": 77.0186,
                },
            },
            done=1,
            n=1,
        )
        assert sample["t_count_judge_s"] == 42.9378
        assert sample["t_yes_no_matcher_s"] == 24.3516
        assert sample["t_policy_generate_s"] == 9.1681
        assert sample["t_search_s"] == 0.4957
        assert sample["t_seed_s"] == 0.0136
        assert sample["t_judges_wall_s"] == 42.9378
        assert sample["timing_breakdown"]["total_s"] == 77.0186

