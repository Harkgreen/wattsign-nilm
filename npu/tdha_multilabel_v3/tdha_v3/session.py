"Every server training entry must run inside a genuinely active tmux session."
import os
import subprocess


def require_tmux():
    message = 'Training and capacity checks require a live tmux session; use bash start.sh or start tmux new -s tdha_v3.'
    if not os.environ.get('TMUX') or not os.environ.get('TMUX_PANE'):
        raise RuntimeError(message)
    try:
        result = subprocess.run(
            ['tmux', 'display-message', '-p', '-t', os.environ['TMUX_PANE'], '#{session_id}'],
            capture_output=True, text=True, timeout=5, check=True,
        )
        if not result.stdout.strip().startswith('$'):
            raise RuntimeError(message)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(message + ' tmux is unavailable or the session is no longer active.') from exc
