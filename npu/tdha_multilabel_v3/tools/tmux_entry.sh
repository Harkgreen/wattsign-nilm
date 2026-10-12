#!/usr/bin/env bash
# Keep the terminal available for inspection after the queue exits; never restart it automatically.
cd -- "$(dirname -- "$0")/.." || exit 2
bash ./start.sh "$@"
tdha_status=$?
printf '\nTDHA queue exited with status %s. Logs and results remain in the configured output directory.\n' "$tdha_status"
printf 'This terminal remains open for inspection. Type exit to close it; run start.sh again to resume.\n'
exec bash --noprofile --norc -i
