"Replace Python with an argument recorder; no training starts and no real NPU is required."
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from tdha_v3.session import require_tmux

ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path('C:/Program Files/Git/bin/bash.exe')
BASH = str(GIT_BASH) if os.name == 'nt' and GIT_BASH.is_file() else shutil.which('bash')
MOCK_TMUX = '''tmux() {
  case "$1" in
    display-message) [ "${TEST_TMUX_STALE:-0}" = 0 ] && printf '$1\\n' ;;
    has-session) [ "${TEST_TMUX_EXISTS:-0}" = 1 ] ;;
    *) printf 'TMUX'; printf '<%s>' "$@"; printf '\\n' ;;
  esac
}; export -f tmux;
'''


@unittest.skipUnless(BASH, 'Bash is unavailable; run launcher tests on the server')
class LauncherTests(unittest.TestCase):
    def launch(self, *args, extra_env=None):
        env = os.environ.copy()
        for key in ('TDHA_DATA', 'TDHA_RUN', 'TDHA_CARDS', 'TDHA_SESSION', 'TDHA_PYTHON', 'TEST_TMUX_EXISTS', 'TEST_TMUX_STALE', 'TDHA_RAM_CACHE_GB', 'TDHA_EVAL_CACHE_GB', 'TDHA_PREFETCH', 'TDHA_REPORT_FONT', 'TEST_PYTHON_FAIL'):
            env.pop(key, None)
        env.update(TMUX='mock-socket,1,0', TMUX_PANE='%0')
        env.update(extra_env or {})
        # Check argument boundaries to ensure paths containing spaces are not split.
        script = MOCK_TMUX + 'python() { printf "MOCK"; printf "<%s>" "$@"; printf "\\n"; if [ "${TEST_PYTHON_FAIL:-}" = "$3" ]; then return 1; fi; }; export -f python; "$BASH" "$@"'
        return subprocess.run([BASH, '-c', script, 'launcher-test', './start.sh', *args],
                              cwd=ROOT, env=env, capture_output=True, encoding='utf-8')

    def test_help_and_invalid_arguments_do_not_run_python(self):
        result = self.launch('--help')
        self.assertEqual(result.returncode, 0)
        self.assertIn('--data', result.stdout)
        self.assertNotIn('MOCK', result.stdout)
        for args in [('--data',), ('--data=',), ('--data', '--devices', '0'), ('--unknown',), ('--session', 'bad:name'), ('--ram-cache-gb','-1'), ('--prefetch','2'), ('--font',), ('--font=',), ('--font','/missing-font.ttf')]:
            result = self.launch(*args)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('MOCK', result.stdout)

    def test_cli_overrides_environment_for_prepare_and_run(self):
        result = self.launch('--data', '/new data/TDHA/V2', '--run', '/new results/pilot', '--devices', '1,2',
                             extra_env={'TDHA_DATA': '/old/data', 'TDHA_RUN': '/old/results', 'TDHA_CARDS': '0'})
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line for line in result.stdout.splitlines() if line.startswith('MOCK')]
        for command in ('prepare', 'run'):
            line = next(line for line in calls if f'<{command}>' in line)
            self.assertIn('<--data></new data/TDHA/V2>', line)
            self.assertIn('<--run></new results/pilot>', line)
        self.assertIn('<--devices><1,2>', next(line for line in calls if '<run>' in line))
        self.assertNotIn('/old/', result.stdout)
        self.assertLess(next(i for i,line in enumerate(calls) if '<preflight>' in line),next(i for i,line in enumerate(calls) if '<prepare>' in line))
        self.assertIn('<--devices><1,2>',next(line for line in calls if '<preflight>' in line))

    def test_defaults_environment_and_equals_syntax(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/datasets/TDHA/V2>', result.stdout)
        result = self.launch(extra_env={'TDHA_DATA': '/env/data'})
        self.assertIn('<--data></env/data>', result.stdout)
        result = self.launch('--data=/equals path/V2', '--devices=2')
        self.assertIn('<--data></equals path/V2>', result.stdout)
        self.assertIn('<--devices><2>', result.stdout)

    def test_relative_path_resolved_before_changing_directory(self):
        # Invoke from the project root and verify that start.sh resolves relative paths inside its own directory.
        script = MOCK_TMUX + 'python() { printf "<%s>" "$@"; printf "\\n"; }; export -f python; "$BASH" tdha_multilabel_v3/start.sh --data "relative data/V2"'
        env = os.environ.copy()
        env.pop('TDHA_DATA', None)
        env.pop('TDHA_PYTHON', None)
        env.update(TMUX='mock-socket,1,0', TMUX_PANE='%0')
        result = subprocess.run([BASH, '-c', script], cwd=ROOT.parent, env=env,
                                capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)
        parent = subprocess.run([BASH, '-c', 'printf "%s" "$PWD"'], cwd=ROOT.parent,
                                capture_output=True, encoding='utf-8', check=True).stdout
        self.assertIn(f'<--data><{parent}/relative data/V2>', result.stdout)
        self.assertNotIn('/tdha_multilabel_v3/relative data/V2>', result.stdout)

    def test_outside_tmux_launches_one_session_without_running_python(self):
        result = self.launch('--data', '/new data/V2', '--session', 'custom-v3',
                             extra_env={'TMUX': '', 'PATH': os.environ['PATH'], 'ASCEND_HOME_PATH': '/custom/cann'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('TMUX<new-session><-d><-s><custom-v3>', result.stdout)
        self.assertIn('/new\\ data/V2', result.stdout)
        self.assertIn('ASCEND_HOME_PATH=/custom/cann', result.stdout)
        self.assertIn('TDHA_PYTHON=', result.stdout)
        self.assertNotIn('MOCK', result.stdout)

    def test_existing_session_does_not_restart_pipeline(self):
        result = self.launch(extra_env={'TMUX': '', 'TEST_TMUX_EXISTS': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('already exists', result.stdout)
        self.assertNotIn('<new-session>', result.stdout)
        self.assertNotIn('MOCK', result.stdout)

    def test_memory_options_forwarded_into_tmux(self):
        result=self.launch('--ram-cache-gb','128','--eval-cache-gb=8','--prefetch','0',extra_env={'TMUX':''})
        self.assertEqual(result.returncode,0,result.stderr)
        for item in ('TDHA_RAM_CACHE_GB=128','TDHA_EVAL_CACHE_GB=8','TDHA_PREFETCH=0'):
            self.assertIn(item,result.stdout)

    def test_skip_preflight_is_explicit_and_forwarded_to_tmux(self):
        normal=self.launch()
        self.assertNotIn('<--skip-preflight>',normal.stdout)
        skipped=self.launch('--skip-preflight')
        self.assertEqual(skipped.returncode,0,skipped.stderr)
        self.assertIn('<preflight>',skipped.stdout)
        self.assertIn('<--skip-preflight>',next(x for x in skipped.stdout.splitlines() if '<preflight>' in x))
        self.assertIn('<prepare>',skipped.stdout);self.assertIn('<run>',skipped.stdout)
        outside=self.launch('--skip-preflight',extra_env={'TMUX':''})
        self.assertEqual(outside.returncode,0,outside.stderr)
        self.assertIn('--skip-preflight',outside.stdout)


    def test_font_is_optional_and_forwarded_with_cli_precedence(self):
        normal=self.launch()
        self.assertEqual(normal.returncode,0,normal.stderr)
        self.assertNotIn('<--font>',normal.stdout)
        self.assertNotIn('<report>',normal.stdout)
        # A real file suffices for this argument test; mocked doctor performs no font parsing.
        supplied=self.launch('--font','./start.sh',extra_env={'TDHA_REPORT_FONT':'/missing-env-font.ttf'})
        self.assertEqual(supplied.returncode,0,supplied.stderr)
        for command in ('doctor','run'):
            line=next(x for x in supplied.stdout.splitlines() if f'<{command}>' in x)
            self.assertIn('<--font>',line)
            self.assertIn('/start.sh>',line)
        self.assertNotIn('missing-env-font',supplied.stdout)
        env_font=self.launch(extra_env={'TDHA_REPORT_FONT':'./start.sh'})
        self.assertEqual(env_font.returncode,0,env_font.stderr)
        self.assertIn('<--font>',env_font.stdout)
        outside=self.launch('--font=./start.sh',extra_env={'TMUX':''})
        self.assertEqual(outside.returncode,0,outside.stderr)
        self.assertIn('TDHA_REPORT_FONT=',outside.stdout)
        self.assertIn('--font',outside.stdout)
        self.assertNotIn('MOCK',outside.stdout)


    def test_failed_queue_packs_without_font_and_forwards_requested_report_font(self):
        normal=self.launch(extra_env={'TEST_PYTHON_FAIL':'run'})
        self.assertEqual(normal.returncode,1,normal.stderr)
        self.assertIn('<pack>',normal.stdout)
        self.assertNotIn('<report>',normal.stdout)
        supplied=self.launch('--font','./start.sh',extra_env={'TEST_PYTHON_FAIL':'run'})
        self.assertEqual(supplied.returncode,1,supplied.stderr)
        report=next(x for x in supplied.stdout.splitlines() if '<report>' in x)
        self.assertIn('<--font>',report)
        self.assertIn('/start.sh>',report)
        self.assertIn('<pack>',supplied.stdout)

    def test_stale_session_stops_before_pipeline(self):
        result = self.launch(extra_env={'TEST_TMUX_STALE': '1'})
        self.assertEqual(result.returncode, 2)
        self.assertNotIn('MOCK', result.stdout)

    def test_missing_tmux_stops_before_pipeline(self):
        script = '''export PATH="/usr/bin:$PATH";
command() { if [ "$1" = -v ] && [ "$2" = tmux ]; then return 1; fi; builtin command "$@"; };
export -f command; "$BASH" ./start.sh'''
        result = subprocess.run([BASH, '-c', script], cwd=ROOT, capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('tmux is unavailable', result.stderr)


class SessionGuardTests(unittest.TestCase):
    def test_requires_live_session(self):
        with patch.dict(os.environ, {}, clear=True), patch('tdha_v3.session.subprocess.run') as call:
            with self.assertRaises(RuntimeError):require_tmux()
            call.assert_not_called()
        with patch.dict(os.environ, {'TMUX': 'socket', 'TMUX_PANE': '%2'}, clear=True):
            with patch('tdha_v3.session.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '$3\n')) as call:
                require_tmux()
                self.assertEqual(call.call_args.args[0][-2:], ['%2', '#{session_id}'])
            for error in (FileNotFoundError(), subprocess.CalledProcessError(1, 'tmux')):
                with patch('tdha_v3.session.subprocess.run', side_effect=error):
                    with self.assertRaises(RuntimeError):require_tmux()

    def test_training_commands_block_before_backend_import(self):
        from tdha_v3.cli import main
        for command in ('preflight', 'run', 'benchmark', 'worker'):
            with patch.dict(os.environ, {}, clear=True), patch('sys.argv', ['tdha_v3', command]):
                with self.assertRaises(SystemExit) as exc:main()
                self.assertEqual(exc.exception.code, 2)
