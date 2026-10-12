from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from load_signature_analysis.data_paths import DataPathError, resolve_data_paths, inspect_file, resolve_project_path
from load_signature_analysis.cli import main, load_result_index

class PublicEntryPointsTests(unittest.TestCase):
    def test_path_precedence_and_no_creation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/"paths.toml").write_text('[paths]\ndata_root="configured"\n',encoding='utf-8')
            paths=resolve_data_paths(project=root,config="paths.toml",data_root="explicit",environ={"NILM_DATA_ROOT":"environment"})
            self.assertEqual(paths.data_root,(root/"explicit").resolve())
            self.assertFalse(paths.output_root.exists())
            self.assertEqual(paths.datasets["ukdale"],(root/"explicit/UK-DALE/ukdale").resolve())

    def test_bounded_file_read_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); data=root/"datasets/REFIT";data.mkdir(parents=True)
            source=data/"example.csv";source.write_bytes(b'time,power\n1,2\n3,4\n5,6\n')
            before=(source.read_bytes(),source.stat().st_mtime_ns)
            paths=resolve_data_paths(project=root,environ={})
            result=inspect_file(paths,"refit","example.csv",max_rows=2,max_bytes=16)
            self.assertLessEqual(result['bytes_read'],16);self.assertEqual(result['rows_read'],2)
            self.assertEqual(before,(source.read_bytes(),source.stat().st_mtime_ns))
            self.assertFalse(paths.output_root.exists())

    def test_traversal_binary_and_raw_output_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);paths=resolve_data_paths(project=root,environ={})
            for filename in ('../outside.csv','archive.zip'):
                with self.assertRaises(DataPathError):inspect_file(paths,'tdha',filename)
            with self.assertRaises(DataPathError):resolve_data_paths(project=root,output_root='datasets/TDHA/V2/cache',environ={})

    def test_missing_result_never_imports_training(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as err:
            with patch('builtins.__import__',wraps=__import__) as imports:
                self.assertEqual(main(['--project-root',td,'results','list']),2)
            self.assertIn('Result index not found:',err.getvalue())
            self.assertIn('results/index.json',err.getvalue().replace('\\','/'))
            self.assertFalse(any('training' in str(call.args[0]) for call in imports.call_args_list))
            self.assertEqual(list(Path(td).iterdir()),[])

    def test_saved_result_index_and_missing_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'results').mkdir();index=root/'results/index.json'
            row=dict(id='tiny',title='Saved evidence',status='completed',summary='An existing measurement.',path='results/tiny')
            index.write_text(json.dumps(dict(schema_version=1,experiments=[row])),encoding='utf-8')
            self.assertEqual(load_result_index(root),[row])
            with redirect_stderr(io.StringIO()),redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--project-root',td,'results','show','tiny']),2)
            row['path']='../outside';index.write_text(json.dumps(dict(schema_version=1,experiments=[row])),encoding='utf-8')
            with self.assertRaises(ValueError):load_result_index(root)

    def test_scientific_paths_honor_dataset_and_output_overrides(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            root=Path(td).resolve();(root/'configs').mkdir()
            (root/'configs/local.toml').write_text('[paths]\ndata_root="raw"\noutput_root="artifacts"\n[datasets]\ntdha="external-tdha"\nukdale="external-ukdale"\n',encoding='utf-8')
            expected={
                'Datasets/TDHA/V2/example.csv':root/'external-tdha/example.csv',
                'datasets/UK-DALE/ukdale/house_1/channel_1.dat':root/'external-ukdale/house_1/channel_1.dat',
                'datasets/REFIT/example.csv':root/'raw/REFIT/example.csv',
                'outputs/runs/demo':root/'artifacts/runs/demo',
                'configs/appliance_taxonomy.yaml':root/'configs/appliance_taxonomy.yaml',
            }
            for original,resolved in expected.items():
                self.assertEqual(resolve_project_path(root,original),resolved)
            with self.assertRaises(DataPathError):resolve_project_path(root,'outputs/../raw/example.csv')
            self.assertEqual(sorted(p.name for p in root.iterdir()),['configs'])
