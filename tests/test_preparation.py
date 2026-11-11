import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class PreparationTests(unittest.TestCase):
    def test_loads_shipped_datasets_from_unrelated_directory(self):
        previous=os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                spec=importlib.util.spec_from_file_location('prepared_data', ROOT/'Prediction Model and GUI'/'DataPrep.py')
                module=importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            finally: os.chdir(previous)
        self.assertEqual(len(module.df_dict),7)
        self.assertEqual(list(module.df_dict),['df_'+name for name in module.event_manifest])
        for key, event_date in zip(module.df_dict, module.event_dates):
            self.assertTrue((module.df_dict[key]['Date'] < event_date).all())
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(module.df_dict[key]['Date']))
        self.assertGreater(len(module.df_other_days),0)
        self.assertFalse(module.df_other_days.isna().any().any())

if __name__ == '__main__': unittest.main()
