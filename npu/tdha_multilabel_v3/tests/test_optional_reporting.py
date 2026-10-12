"""Mocked queue/report contracts; no fitting, prediction or hardware is invoked."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tdha_v3.cli import main
from tdha_v3.common import Blocked, dump, read
from tdha_v3.reporting import report, pack


class OptionalReportingTests(unittest.TestCase):
    def test_no_font_run_packs_completed_results_without_rendering(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run'
            def completed(*args):
                dump(run/'execution_complete.json',{'complete':True})
            def analyzed(*args):dump(run/'analysis/decision.json',{'summary':'mocked frozen analysis'})
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.runtime.run_queue',side_effect=completed) as queue,patch('tdha_v3.analysis.analyze',side_effect=analyzed) as analyze,patch('tdha_v3.reporting.report') as render,patch('tdha_v3.reporting.validate_report_font') as font:
                self.assertEqual(main(),0)
            queue.assert_called_once()
            analyze.assert_called_once_with(run)
            render.assert_not_called()
            font.assert_not_called()
            self.assertFalse((run/'report').exists())
            with zipfile.ZipFile(run/'downloads/TDHA_V3_报告与汇总.zip') as archive:
                self.assertIn('execution_complete.json',archive.namelist())
                self.assertIn('analysis/decision.json',archive.namelist())

    def test_requested_font_validates_before_queue_then_renders_and_packs(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';font=Path(temp)/'font with spaces.ttf'
            font.write_bytes(b'font placeholder for mocked validation')
            calls=[]
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run),'--font',str(font)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.reporting.validate_report_font',side_effect=lambda *a:calls.append('validate') or font),patch('tdha_v3.runtime.run_queue',side_effect=lambda *a:calls.append('queue')),patch('tdha_v3.reporting.report',side_effect=lambda *a:calls.append('report')) as render,patch('tdha_v3.reporting.pack',side_effect=lambda *a:calls.append('pack')):
                self.assertEqual(main(),0)
            self.assertEqual(calls,['validate','queue','report','pack'])
            self.assertEqual(render.call_args.args,(run,font.resolve()))

    def test_missing_font_and_rendering_dependency_stop_without_output_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);font=root/'font.ttf';font.write_bytes(b'font placeholder')
            for command,extra,validation in [
                ('report',[],None),
                ('run',['--font',str(root/'missing.ttf')],None),
                ('run',['--font',str(font)],ModuleNotFoundError('reportlab')),
            ]:
                with self.subTest(command=command,extra=extra):
                    run=root/('output-'+str(len(extra))+'-'+command)
                    with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3',command,'--run',str(run),*extra]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.runtime.run_queue') as queue,patch('tdha_v3.reporting.validate_report_font',side_effect=validation):
                        with self.assertRaises(SystemExit) as exc:main()
                        self.assertEqual(exc.exception.code,2)
                    queue.assert_not_called()
                    self.assertFalse(run.exists())

    def test_direct_report_without_font_creates_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'new-run'
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}):
                with self.assertRaisesRegex(Blocked,'requires --font'):report(run)
            self.assertFalse(run.exists())

    def test_render_failure_still_packs_without_marking_completed_training_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';font=Path(temp)/'font.ttf';font.write_bytes(b'mocked font')
            def completed(*args):dump(run/'execution_complete.json',{'complete':True})
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run),'--font',str(font)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.reporting.validate_report_font',return_value=font),patch('tdha_v3.runtime.run_queue',side_effect=completed),patch('tdha_v3.analysis.analyze'),patch('tdha_v3.reporting.report',side_effect=RuntimeError('renderer failure')):
                self.assertEqual(main(),1)
            self.assertFalse((run/'blocker.json').exists())
            errors=list((run/'audit/errors').glob('report-*.json'))
            self.assertEqual(len(errors),1)
            self.assertEqual(read(errors[0])['command'],'report')
            with zipfile.ZipFile(run/'downloads/TDHA_V3_报告与汇总.zip') as archive:
                self.assertIn('execution_complete.json',archive.namelist())

    def test_existing_decision_is_reused_and_partial_queue_does_not_analyze(self):
        with tempfile.TemporaryDirectory() as temp:
            for completed in (False,True):
                with self.subTest(completed=completed):
                    run=Path(temp)/str(completed);run.mkdir()
                    if completed:
                        dump(run/'execution_complete.json',{'complete':True})
                        dump(run/'analysis/decision.json',{'summary':'existing frozen analysis'})
                    with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.runtime.run_queue'),patch('tdha_v3.analysis.analyze') as analyze,patch('tdha_v3.reporting.report') as render:
                        self.assertEqual(main(),0)
                    analyze.assert_not_called()
                    render.assert_not_called()
                    self.assertTrue((run/'downloads/TDHA_V3_报告与汇总.zip').is_file())

    def test_nonrendered_pack_contains_saved_metrics_without_weights_or_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp);task=run/'tasks/N0-W1-A-s42'
            for name in ('done.json','readout.json','thresholds.json','time_metrics.json','cross_metrics.json','memory.json'):
                dump(task/name,{'saved':name})
            (task/'time_metrics.csv').write_text('AP,Recall\n0.9,0.8\n',encoding='utf-8')
            for name in ('best.pt','time.npy','last.pt'):(task/name).write_bytes(b'excluded')
            pack(run)
            with zipfile.ZipFile(run/'downloads/TDHA_V3_报告与汇总.zip') as archive:
                names=set(archive.namelist())
                for name in ('done.json','readout.json','thresholds.json','time_metrics.json','cross_metrics.json','memory.json','time_metrics.csv'):
                    entry='tasks/N0-W1-A-s42/'+name
                    self.assertIn(entry,names)
                    self.assertEqual(archive.read(entry),(task/name).read_bytes())
                self.assertFalse(any(n.endswith(('.pt','.npy')) for n in names))


    def test_analysis_failure_still_packs_with_correct_error_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run'
            def completed(*args):dump(run/'execution_complete.json',{'complete':True})
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.runtime.run_queue',side_effect=completed),patch('tdha_v3.analysis.analyze',side_effect=RuntimeError('analysis failure')),patch('tdha_v3.reporting.report') as render:
                self.assertEqual(main(),1)
            render.assert_not_called()
            self.assertEqual(read(run/'blocker.json')['command'],'analysis')
            self.assertTrue((run/'downloads/TDHA_V3_报告与汇总.zip').is_file())

    def test_report_and_pack_failures_keep_original_error_and_separate_pack_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run';font=Path(temp)/'font.ttf';font.write_bytes(b'mocked font')
            def completed(*args):dump(run/'execution_complete.json',{'complete':True})
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run),'--font',str(font)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.reporting.validate_report_font',return_value=font),patch('tdha_v3.runtime.run_queue',side_effect=completed),patch('tdha_v3.analysis.analyze'),patch('tdha_v3.reporting.report',side_effect=RuntimeError('original renderer failure')),patch('tdha_v3.reporting.pack',side_effect=OSError('secondary packing failure')):
                self.assertEqual(main(),1)
            reports=list((run/'audit/errors').glob('report-*.json'))
            packs=list((run/'audit/errors').glob('pack-*.json'))
            self.assertEqual(len(reports),1);self.assertEqual(len(packs),1)
            self.assertEqual(read(reports[0])['reason'],'original renderer failure')
            self.assertEqual(read(packs[0])['reason'],'secondary packing failure')
            self.assertFalse((run/'blocker.json').exists())

    def test_pack_only_failure_is_recorded_as_pack(self):
        with tempfile.TemporaryDirectory() as temp:
            run=Path(temp)/'run'
            with patch.dict(os.environ,{'TDHA_REPORT_FONT':''}),patch('sys.argv',['tdha_v3','run','--run',str(run)]),patch('tdha_v3.session.require_tmux'),patch('tdha_v3.runtime.run_queue'),patch('tdha_v3.reporting.pack',side_effect=OSError('packing failure')):
                self.assertEqual(main(),1)
            errors=list((run/'audit/errors').glob('pack-*.json'))
            self.assertEqual(len(errors),1)
            self.assertEqual(read(errors[0])['command'],'pack')
            self.assertEqual(read(run/'blocker.json')['command'],'pack')
