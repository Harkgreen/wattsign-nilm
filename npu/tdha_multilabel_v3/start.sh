#!/usr/bin/env bash
set -euo pipefail
TDHA_RUN="${TDHA_RUN:-outputs/tdha_v3/pilot_v3}"
TDHA_DATA="${TDHA_DATA:-datasets/TDHA/V2}"
TDHA_CARDS="${TDHA_CARDS:-0,1,2}"
TDHA_SESSION="${TDHA_SESSION:-tdha_v3}"
TDHA_REPORT_FONT="${TDHA_REPORT_FONT:-}"
TDHA_SKIP_PREFLIGHT=0
export TDHA_RAM_CACHE_GB="${TDHA_RAM_CACHE_GB:-64}"
export TDHA_EVAL_CACHE_GB="${TDHA_EVAL_CACHE_GB:-4}"
export TDHA_PREFETCH="${TDHA_PREFETCH:-1}"
usage() {
  printf '%s\n' \
    'Usage: bash start.sh [--data DATA_DIR] [--run OUTPUT_DIR] [--devices IDS] [--session NAME]' \
    'Runs the full exploratory training queue, including candidate selection, fine-tuning and independent replays.' \
    'Memory: --ram-cache-gb 64 (GiB per process) --eval-cache-gb 4 (GiB per NPU) --prefetch 1 (0 disables)' \
    'Optional report export: --font LICENSED_CJK_TTF, or TDHA_REPORT_FONT. Without a font, only results are packed.' \
    'Record a skipped operator preflight: --skip-preflight (dependency, data, capacity and reproducibility checks remain).' \
    'Example: bash start.sh --data "/data/TDHA/V2" --run "/outputs/tdha_v3" --devices 0,1,2' \
    'The data directory must directly contain Aggregate.zip, SocketRecord.xlsx and individual-appliance ZIP files.' \
    'Precedence: command-line options > TDHA_* environment variables > defaults.' \
    'Relative paths are resolved from the calling directory. Quote paths containing spaces.' \
    'Starts in tmux. An existing session is reattached without launching another queue.' \
    'An active tmux session runs the queue directly. Reuse the same --run directory to resume.'
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --skip-preflight) TDHA_SKIP_PREFLIGHT=1; shift ;;
    --data|--run|--devices|--session|--ram-cache-gb|--eval-cache-gb|--prefetch|--font)
      if [ "$#" -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
        printf 'Error: %s requires a value.\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --data) TDHA_DATA="$2" ;;
        --run) TDHA_RUN="$2" ;;
        --devices) TDHA_CARDS="$2" ;;
        --session) TDHA_SESSION="$2" ;;
        --ram-cache-gb) TDHA_RAM_CACHE_GB="$2" ;;
        --eval-cache-gb) TDHA_EVAL_CACHE_GB="$2" ;;
        --prefetch) TDHA_PREFETCH="$2" ;;
        --font) TDHA_REPORT_FONT="$2" ;;
      esac
      shift 2 ;;
    --data=*|--run=*|--devices=*|--session=*|--ram-cache-gb=*|--eval-cache-gb=*|--prefetch=*|--font=*)
      tdha_value="${1#*=}"
      if [ -z "$tdha_value" ]; then
        printf 'Error: %s cannot be empty.\n' "${1%%=*}" >&2
        exit 2
      fi
      case "$1" in
        --data=*) TDHA_DATA="$tdha_value" ;;
        --run=*) TDHA_RUN="$tdha_value" ;;
        --devices=*) TDHA_CARDS="$tdha_value" ;;
        --session=*) TDHA_SESSION="$tdha_value" ;;
        --ram-cache-gb=*) TDHA_RAM_CACHE_GB="$tdha_value" ;;
        --eval-cache-gb=*) TDHA_EVAL_CACHE_GB="$tdha_value" ;;
        --prefetch=*) TDHA_PREFETCH="$tdha_value" ;;
        --font=*) TDHA_REPORT_FONT="$tdha_value" ;;
      esac
      shift ;;
    *) printf 'Error: unknown option %s; use --help for usage.\n' "$1" >&2; exit 2 ;;
  esac
done
if [[ ! "$TDHA_RAM_CACHE_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ ! "$TDHA_EVAL_CACHE_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ ! "$TDHA_PREFETCH" =~ ^[01]$ ]]; then
  printf 'Error: cache capacities must be nonnegative and prefetch must be 0 or 1.\n' >&2
  exit 2
fi
if [[ ! "$TDHA_SESSION" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  printf 'Error: session names accept only letters, digits, underscores and hyphens.\n' >&2
  exit 2
fi
# Resolve paths before changing directories or launching a tmux session.
case "$TDHA_DATA" in /*) ;; *) TDHA_DATA="$PWD/$TDHA_DATA" ;; esac
case "$TDHA_RUN" in /*) ;; *) TDHA_RUN="$PWD/$TDHA_RUN" ;; esac
if [ -n "$TDHA_REPORT_FONT" ]; then
  case "$TDHA_REPORT_FONT" in /*) ;; *) TDHA_REPORT_FONT="$PWD/$TDHA_REPORT_FONT" ;; esac
  if [ ! -f "$TDHA_REPORT_FONT" ] || [ ! -r "$TDHA_REPORT_FONT" ]; then
    printf 'Error: report font is missing or unreadable: %s\n' "$TDHA_REPORT_FONT" >&2
    exit 2
  fi
  export TDHA_REPORT_FONT
fi
cd -- "$(dirname -- "$0")"
if ! command -v tmux >/dev/null 2>&1; then
  printf 'Error: tmux is unavailable; install it before starting the queue.\n' >&2
  exit 2
fi
if [ -n "${TMUX:-}" ]; then
  if [ -z "${TMUX_PANE:-}" ] || ! tmux display-message -p -t "$TMUX_PANE" '#{session_id}' >/dev/null 2>&1; then
    printf 'Error: the tmux environment is stale; use a live session or a new terminal.\n' >&2
    exit 2
  fi
else
  if tmux has-session -t "=$TDHA_SESSION" 2>/dev/null; then
    printf 'Session %s already exists. These options will not launch a new queue.\n' "$TDHA_SESSION"
  else
    # Forward the active conda/CANN environment rather than the tmux server environment.
    tdha_env=(env "TDHA_PYTHON=${TDHA_PYTHON:-$(command -v python)}")
    for tdha_key in PATH LD_LIBRARY_PATH PYTHONPATH LD_PRELOAD CONDA_PREFIX CONDA_DEFAULT_ENV ASCEND_RT_VISIBLE_DEVICES; do
      if [[ -v "$tdha_key" ]]; then tdha_env+=("$tdha_key=${!tdha_key}"); fi
    done
    while IFS= read -r tdha_key; do
      case "$tdha_key" in ASCEND_*|CANN_*|HCCL_*|NPU_*|TDHA_RAM_*|TDHA_EVAL_*|TDHA_HBM_*|TDHA_PREFETCH|TDHA_REPORT_FONT) tdha_env+=("$tdha_key=${!tdha_key}");; esac
    done < <(compgen -e)
    tdha_args=(--data "$TDHA_DATA" --run "$TDHA_RUN" --devices "$TDHA_CARDS" --session "$TDHA_SESSION")
    if [ "$TDHA_SKIP_PREFLIGHT" = 1 ]; then tdha_args+=(--skip-preflight); fi
    if [ -n "$TDHA_REPORT_FONT" ]; then tdha_args+=(--font "$TDHA_REPORT_FONT"); fi
    printf -v tdha_command '%q ' "${tdha_env[@]}" bash "$PWD/tools/tmux_entry.sh" "${tdha_args[@]}"
    tmux new-session -d -s "$TDHA_SESSION" -c "$PWD" "$tdha_command"
    printf 'Started tmux session %s. The queue continues after an SSH disconnect.\n' "$TDHA_SESSION"
  fi
  printf 'Reconnect: tmux attach -t %s\nDetach: press Ctrl+B, release, then D.\n' "$TDHA_SESSION"
  if [ -t 0 ] && [ -t 1 ]; then exec tmux attach-session -t "=$TDHA_SESSION"; fi
  exit 0
fi
TDHA_PYTHON="${TDHA_PYTHON:-python}"
printf 'Data: %s\nOutput: %s\nNPUs: %s\n' "$TDHA_DATA" "$TDHA_RUN" "$TDHA_CARDS"
printf 'RAM cache per process: %s GiB; evaluation cache per NPU: %s GiB; CPU prefetch: %s\n' "$TDHA_RAM_CACHE_GB" "$TDHA_EVAL_CACHE_GB" "$TDHA_PREFETCH"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
tdha_report_args=()
if [ -n "$TDHA_REPORT_FONT" ]; then tdha_report_args+=(--font "$TDHA_REPORT_FONT"); fi
failure_report() {
  tdha_exit=$?
  trap - ERR
  if [ -n "$TDHA_REPORT_FONT" ]; then
    "$TDHA_PYTHON" -m tdha_v3 report --run "$TDHA_RUN" "${tdha_report_args[@]}" || true
  fi
  "$TDHA_PYTHON" -m tdha_v3 pack --run "$TDHA_RUN" || true
  exit "$tdha_exit"
}
trap failure_report ERR
"$TDHA_PYTHON" -m tdha_v3 doctor --run "$TDHA_RUN" "${tdha_report_args[@]}"
"$TDHA_PYTHON" -m tdha_v3 test
tdha_preflight_args=()
if [ "$TDHA_SKIP_PREFLIGHT" = 1 ]; then tdha_preflight_args+=(--skip-preflight); fi
"$TDHA_PYTHON" -m tdha_v3 preflight --run "$TDHA_RUN" --devices "$TDHA_CARDS" "${tdha_preflight_args[@]}"
"$TDHA_PYTHON" -m tdha_v3 prepare --run "$TDHA_RUN" --data "$TDHA_DATA"
if [ ! -f "$TDHA_RUN/benchmark.json" ]; then
  "$TDHA_PYTHON" -m tdha_v3 benchmark --run "$TDHA_RUN" --devices "$TDHA_CARDS"
fi
if ! "$TDHA_PYTHON" -m tdha_v3 run --run "$TDHA_RUN" --data "$TDHA_DATA" --devices "$TDHA_CARDS" --resume "${tdha_report_args[@]}"; then
  if [ -n "$TDHA_REPORT_FONT" ]; then
    "$TDHA_PYTHON" -m tdha_v3 report --run "$TDHA_RUN" "${tdha_report_args[@]}" || true
  fi
  "$TDHA_PYTHON" -m tdha_v3 pack --run "$TDHA_RUN" || true
  exit 1
fi
