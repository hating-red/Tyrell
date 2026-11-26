import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from diagnostic_ablation import run_ablations


class JoinAblationTests(unittest.TestCase):
    def run_trial(self, control, trial, seconds=100, joins=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            runtime = path / "runtime.json"
            runtime.write_text(json.dumps({"join_audit": joins or [
                {"table": "history", "status": "applied", "output_columns": ["history__value"]},
                {"table": "hourly", "status": "skipped", "reason": "unknown availability"}]}))
            experiment = Mock(return_value=(trial, {}))
            with patch.dict(os.environ, {"TYRELL_RUNTIME_STATS": str(runtime), "TYRELL_OUTPUT_DIR": tmp}):
                run_ablations(path, {"metric": "mae"}, control, time.monotonic() + seconds, experiment)
            result = json.loads((path / "join-ablations.json").read_text(encoding="utf-8"))
        return result["groups"], experiment

    def setUp(self):
        self.control = dict(score=2., complete=True, fit_time=1., split={"type": "time"},
                            fold_scores=[1., 3.], fold_signature="abc", greater_is_better=False)

    def test_gain_is_positive_when_removing_lookup_hurts_mae(self):
        trial = self.control | {"score": 2.5, "fold_scores": [2., 3.]}
        (group, skipped), spy = self.run_trial(self.control, trial)
        self.assertEqual(group["fold_gains_with_group"], [1., 0.])
        self.assertEqual(group["gain_with_group"], .5)
        self.assertEqual(skipped["reason"], "unknown availability")
        self.assertEqual(spy.call_args.kwargs["config"],
                         {"exclude_columns": ["history__value"], "diagnostic_group": "history"})

    def test_unmatched_or_missing_control_is_not_compared(self):
        (group, _), _ = self.run_trial(self.control, self.control | {"fold_signature": "other"})
        self.assertEqual(group["status"], "incomparable_folds")
        (group, _), spy = self.run_trial(None, None)
        self.assertEqual(group["status"], "no_complete_temporal_control")
        spy.assert_not_called()

    def test_small_budget_skips_experiment(self):
        (group, _), spy = self.run_trial(self.control, None, seconds=.01)
        self.assertEqual(group["status"], "insufficient_budget")
        spy.assert_not_called()

    def test_delayed_training_membership_must_match(self):
        control = self.control | {"membership_signature": "train-A"}
        result = self.control | {"membership_signature": "train-B"}
        (group, _), _ = self.run_trial(control, result)
        self.assertEqual(group["status"], "incomparable_folds")

    def test_parent_ablation_also_removes_chained_features(self):
        joins = [{"table": "s.creative", "status": "applied", "output_columns": ["creative__owner"]},
                 {"table": "s.owner", "status": "applied", "output_columns": ["owner__kind"],
                  "dependencies": ["s.creative"]}]
        (parent, child), spy = self.run_trial(self.control, self.control, joins=joins)
        self.assertEqual(parent["dependent_tables"], ["s.owner"])
        self.assertEqual(parent["columns"], ["creative__owner", "owner__kind"])
        self.assertEqual(child["columns"], ["owner__kind"])
        self.assertEqual(spy.call_args_list[0].kwargs["config"]["diagnostic_group"], "s.owner")
