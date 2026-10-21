import unittest

from pde_agent_verifier_solver.plugins.ns.publish import _primary_error


class NSPublishTests(unittest.TestCase):
    def test_customer_result_prefers_hidden_error(self) -> None:
        summary = {
            "generated_summary": {"mean_relative_l2": 0.5},
            "hidden_holdout_summary": {"mean_relative_l2": 0.25},
        }

        self.assertEqual(_primary_error(summary), 0.25)

    def test_customer_result_prefers_complete_accumulated_hidden_suite(self) -> None:
        summary = {
            "hidden_all_summary": {"mean_relative_l2": 0.4},
            "hidden_generated_cumulative_summary": {"mean_relative_l2": 0.3},
            "hidden_holdout_summary": {"mean_relative_l2": 0.2},
            "generated_summary": {"mean_relative_l2": 0.1},
        }

        self.assertEqual(_primary_error(summary), 0.4)


if __name__ == "__main__":
    unittest.main()
