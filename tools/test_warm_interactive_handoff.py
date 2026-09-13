"""Exercise the runner's handoff dispatch without importing Isaac's app.

The runner is an executable script with simulator startup at module scope;
extract the small dispatch branch from its AST and execute that exact code.
"""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


class WarmHandoffTests(unittest.TestCase):
    def dispatch(self, supported):
        source = Path(__file__).resolve().parents[1] / 'run_motion_pipeline_sim.py'
        tree = ast.parse(source.read_text())
        candidates = [node for node in ast.walk(tree)
                      if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Name)
                      and node.test.id == 'support_active'
                      and 'provider.begin_control_handoff()' in ast.unparse(node)
                      and 'provider.end_control_handoff()' in ast.unparse(node)]
        self.assertEqual(len(candidates), 1)
        provider = Mock()
        branch = ast.Module(body=[candidates[0]], type_ignores=[])
        exec(compile(branch, str(source), 'exec'),
             {'provider': provider, 'support_active': supported})
        return provider

    def test_supported_start_keeps_bootstrap_handoff(self):
        provider = self.dispatch(True)
        provider.begin_control_handoff.assert_called_once_with()
        provider.end_control_handoff.assert_not_called()

    def test_warm_return_does_not_rearm_bootstrap(self):
        provider = self.dispatch(False)
        provider.begin_control_handoff.assert_not_called()
        provider.end_control_handoff.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
