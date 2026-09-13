"""Check deployment solver defaults and bounded diagnostic overrides."""
import ast
import os
from pathlib import Path
import unittest
from unittest.mock import patch


class SolverIterationTests(unittest.TestCase):
    def resolve(self, **values):
        source = Path(__file__).resolve().parents[1] / (
            'tasks/g1_tasks/flat_ground_g1_29dof_deployment_v1/'
            'flat_ground_g1_29dof_deployment_v1_env_cfg.py')
        helper = next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n, ast.FunctionDef) and n.name == '_solver_iterations')
        namespace = {'os': os}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
        with patch.dict(os.environ, values, clear=True):
            return namespace['_solver_iterations']()

    def test_defaults(self):
        self.assertEqual(self.resolve(), (8, 1))

    def test_previous_eight_four_configuration_remains_available(self):
        self.assertEqual(self.resolve(G1_DEPLOYMENT_SOLVER_VELOCITY_ITERS='4'), (8, 4))

    def test_baseline_override(self):
        self.assertEqual(self.resolve(G1_DEPLOYMENT_SOLVER_POSITION_ITERS='4',
                                      G1_DEPLOYMENT_SOLVER_VELOCITY_ITERS='1'), (4, 1))

    def test_outside_bounds_rejected(self):
        for key, value in [('POSITION', '3'), ('POSITION', '9'),
                           ('VELOCITY', '0'), ('VELOCITY', '5')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.resolve(**{f'G1_DEPLOYMENT_SOLVER_{key}_ITERS': value})


if __name__ == '__main__':
    unittest.main()
