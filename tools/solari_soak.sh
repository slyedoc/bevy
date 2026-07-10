#!/usr/bin/env bash
# Solari cold-start soak: launch a solari exam N times and fail on any tripwire.
#
# Every cold-start bug in this codebase's history was timing-dependent —
# invisible in any single green run (async pipeline-compile order, task-pool
# scheduling, sparse-bind races). This soak makes those regressions visible the
# day they land: repeated fresh launches, each scanned for the failure
# signatures those bugs taught us (see crates/bevy_solari/src/ecs_gpu/
# readiness.rs for the class rules).
#
# Usage:
#   tools/solari_soak.sh [RUNS]            # default 10 furnace launches
#   SOAK_CMD="..." tools/solari_soak.sh 5  # soak any command instead (e.g. the
#                                          # solari_files viewer); it must exit
#                                          # by itself (CLAUDECODE-style timeout
#                                          # or AUTO_* env) — the outer timeout
#                                          # here is only a hang backstop.
#
# Exit code: 0 = all runs clean; 1 = at least one tripwire (failing logs kept).

set -u

RUNS="${1:-10}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$(mktemp -d /tmp/solari_soak.XXXXXX)"
FEATURES="bevy_solari bevy_solari_debug cluster_processor free_camera bevy_feathers bevy_dev_tools"
# Rotate scenes so the soak exercises different startup workloads (emissive
# grid vs sky furnace vs sun+lamps).
SCENES=(furnace room lamps yard)
# Per-run hang backstop (seconds). The app self-exits at -t 6; if the outer
# timeout fires, the run failed by definition (hang / lost AppExit).
BACKSTOP=45

cd "$REPO_ROOT"

if [[ -z "${SOAK_CMD:-}" ]]; then
    echo "soak: building solari_furnace..."
    if ! cargo build --example solari_furnace --features "$FEATURES" 2>&1 | tail -3; then
        echo "soak: BUILD FAILED"
        exit 1
    fi
fi

# Kernel-log watermark: any NVIDIA Xid logged during the soak window is a
# device-level fault (pushbuffer corruption, page fault) even if the app's own
# log looks clean. Best-effort — skipped when journalctl isn't readable.
JOURNAL_SINCE="$(date '+%Y-%m-%d %H:%M:%S')"

fail_count=0
for i in $(seq 1 "$RUNS"); do
    log="$LOG_DIR/run$i.log"
    if [[ -n "${SOAK_CMD:-}" ]]; then
        timeout -k 10 "$BACKSTOP" bash -c "$SOAK_CMD" >"$log" 2>&1
    else
        scene="${SCENES[$(((i - 1) % ${#SCENES[@]}))]}"
        timeout -k 10 "$BACKSTOP" ./target/debug/examples/solari_furnace \
            -t 6 --scene "$scene" >"$log" 2>&1
    fi
    exit_code=$?

    # ── Tripwires ────────────────────────────────────────────────────────────
    # Hard failures: crashes, device loss, validation errors, leaked objects,
    # probe FAILs, and the one-shot-loss recovery/warn lines (their firing means
    # a producer/consumer readiness hole re-opened — see readiness.rs).
    trips=$(grep -cE \
        "panicked at|DEVICE_LOST|VALIDATION|leaked object|frontier: bail|unwalked .* re-armed|full walk re-armed|no column resource|clas: DROPPING|GPU checkpoints|probe\(s\) FAILED" \
        "$log")
    # The PTLAS null-AS heal firing once at warmup is tolerated; repeats mean
    # the rebuild-until-clean loop is stuck (transforms never became ready).
    nulls=$(grep -c "null-AS" "$log")

    verdict="ok"
    reasons=()
    [[ $exit_code -ne 0 ]] && reasons+=("exit=$exit_code")
    [[ $trips -gt 0 ]] && reasons+=("$trips tripwire line(s)")
    [[ $nulls -gt 2 ]] && reasons+=("null-AS heal looped ($nulls)")
    if [[ ${#reasons[@]} -gt 0 ]]; then
        verdict="FAIL: ${reasons[*]}"
        fail_count=$((fail_count + 1))
    fi
    echo "soak run $i/$RUNS: $verdict"
    if [[ $verdict != ok ]]; then
        grep -m 5 -E \
            "panicked at|DEVICE_LOST|VALIDATION|leaked object|frontier: bail|re-armed|no column resource|DROPPING|FAILED" \
            "$log" | sed 's/^/    /'
    fi
done

# Kernel Xids during the window (best-effort; needs journal read access).
if xids=$(journalctl -k --since "$JOURNAL_SINCE" --no-pager 2>/dev/null | grep -i "xid"); then
    if [[ -n "$xids" ]]; then
        echo "soak: KERNEL Xid(s) during the soak window:"
        echo "$xids" | sed 's/^/    /'
        fail_count=$((fail_count + 1))
    fi
fi

if [[ $fail_count -gt 0 ]]; then
    echo "soak: $fail_count failure(s); logs kept in $LOG_DIR"
    exit 1
fi
echo "soak: all $RUNS runs clean"
rm -rf "$LOG_DIR"
exit 0
