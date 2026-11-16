import os
import unittest
from unittest.mock import patch


class ProcessingProfileTests(unittest.TestCase):
    def test_catalog_supplies_detached_valid_inference_options(self):
        from processing_profiles import profile_catalog, profile_defaults
        from inference_options import InferenceOptions
        import json
        catalog = profile_catalog()
        self.assertEqual([item['id'] for item in catalog], ['fast', 'balanced', 'quality'])
        for item in catalog:
            self.assertTrue(item['label'])
            self.assertTrue(item['description'])
            self.assertEqual(item['options']['preset'], item['id'])
            self.assertEqual(InferenceOptions.parse(json.dumps(item['options'])).preset, item['id'])
        catalog[0]['options']['steps'] = 24
        self.assertEqual(profile_defaults('fast')['steps'], 5)
        defaults = profile_defaults('balanced')
        defaults['lora_scale'] = 2
        self.assertEqual(profile_defaults('balanced')['lora_scale'], 1)
        with self.assertRaises(ValueError):
            profile_defaults('custom')

    def test_environment_default_is_normalized_and_invalid_values_fall_back(self):
        from processing_profiles import default_profile
        for value, expected in ((None, 'balanced'), ('', 'balanced'), (' FAST ', 'fast'),
                                ('quality', 'quality'), ('wrong', 'balanced'), ('custom', 'balanced')):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ['CLEAR_PRESET'] = value
                self.assertEqual(default_profile(), expected)
