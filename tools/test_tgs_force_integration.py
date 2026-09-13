"""Test the exact scene-authoring and profile helpers without starting Isaac Sim."""
import ast
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


class TgsForceIntegrationTests(unittest.TestCase):
    def resolve(self, profile, value=None):
        source = Path(__file__).resolve().parents[1] / 'run_motion_pipeline_sim.py'
        tree = ast.parse(source.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'resolve_tgs_force_integration')
        namespace = {'os': os}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
        overrides = {} if value is None else {'SONIC_TGS_FORCES_EVERY_ITERATION': value}
        with patch.dict(os.environ, overrides, clear=True):
            return namespace['resolve_tgs_force_integration'](profile)

    def test_default_is_scoped_to_deployment_profile(self):
        self.assertTrue(self.resolve('g1_deployment_v1'))
        self.assertFalse(self.resolve('sonic_official_g1'))

    def test_explicit_override_and_invalid_value(self):
        self.assertFalse(self.resolve('g1_deployment_v1', '0'))
        self.assertTrue(self.resolve('sonic_official_g1', '1'))
        with self.assertRaises(ValueError):
            self.resolve('g1_deployment_v1', 'true')

    def configure(self, enabled, existing_stage):
        source = Path(__file__).resolve().parents[1] / 'run_motion_pipeline_sim.py'
        tree = ast.parse(source.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'configure_tgs_force_integration')
        namespace = {}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
        names = ['isaacsim', 'isaacsim.core', 'isaacsim.core.utils',
                 'isaacsim.core.utils.stage', 'pxr']
        modules = {name: ModuleType(name) for name in names}
        for name in names:
            if '.' in name:
                parent, child = name.rsplit('.', 1)
                setattr(modules[parent], child, modules[name])
        stage = modules['isaacsim.core.utils.stage']
        stage.get_current_stage = Mock(return_value=existing_stage)
        stage.create_new_stage = Mock(return_value=object())
        pxr = modules['pxr']
        pxr.UsdPhysics = Mock()
        pxr.PhysxSchema = Mock()
        with patch.dict(sys.modules, modules):
            namespace['configure_tgs_force_integration']('/physicsScene', enabled)
        expected = existing_stage if existing_stage is not None else stage.create_new_stage.return_value
        pxr.UsdPhysics.Scene.Define.assert_called_once_with(expected, '/physicsScene')
        pxr.PhysxSchema.PhysxSceneAPI.Apply.assert_called_once_with(
            pxr.UsdPhysics.Scene.Define.return_value.GetPrim.return_value)
        pxr.PhysxSchema.PhysxSceneAPI.Apply.return_value.CreateEnableExternalForcesEveryIterationAttr.assert_called_once_with(enabled)
        return stage

    def test_enable_on_existing_stage(self):
        self.configure(True, object()).create_new_stage.assert_not_called()

    def test_create_missing_stage(self):
        self.configure(True, None).create_new_stage.assert_called_once_with()

    def test_explicit_disable_for_comparison(self):
        self.configure(False, object()).create_new_stage.assert_not_called()


if __name__ == '__main__':
    unittest.main()
