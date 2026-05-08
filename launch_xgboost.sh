#!/bin/bash
# Sweep hyperparameter combinations across multiple evaluation windows.
# Results are collected into logs/summary.csv for easy comparison.
#
# Usage:
#   bash sweep_params.sh

set -euo pipefail

LOG_DIR="logs2"
SUMMARY="${LOG_DIR}/summary.csv"
mkdir -p "${LOG_DIR}"

# hyperparameters to sweep
SUBSAMPLES=(0.5 0.8)
COLSAMPLES=(0.3 0.8)
MIN_CHILD_WEIGHTS=(10 50)
TOP_K=(38 50)

# as-of dates and their corresponding test windows
# format: "as_of|test_start|test_end|label"
WINDOWS=(
    "20260424|20260428|20260430|apr2026"
    "20260107|20260108|20260112|jan2026"
    "20251119|20251120|20251124|nov2025"
    "20250910|20250911|20250915|sep2025"
)

# write CSV header
echo "subsample,colsample,min_child_weight,k,window,as_of,test_start,test_end,portfolio_return,benchmark_return,excess_return,rank_ic" > "${SUMMARY}"

for k in "${TOP_K[@]}"; do
    for ss in "${SUBSAMPLES[@]}"; do
        for cs in "${COLSAMPLES[@]}"; do
            for mcw in "${MIN_CHILD_WEIGHTS[@]}"; do
                tag="ss${ss}_cs${cs}_mcw${mcw}_k${k}"
                mkdir -p "${LOG_DIR}/${tag}"

                for window_spec in "${WINDOWS[@]}"; do
                    # parse the window spec
                    IFS='|' read -r as_of test_start test_end label <<< "${window_spec}"

                    log_file="${LOG_DIR}/${tag}/${label}.log"
                    submission="temp_${tag}_${label}.csv"

                    echo "=== ${tag} | window=${label} ==="

                    # generate portfolio
                    train_output=$(python -u tweak_baseline.py \
                        --as-of "${as_of}" \
                        --top-k "${k}" \
                        --subsample "${ss}" \
                        --colsample "${cs}" \
                        --min-child-weight "${mcw}" \
                        --out "${submission}" \
                        2>&1 | tee "${log_file}")
                    
                    # extract IC from tagged line
                    rank_ic=$(echo "${train_output}" | grep "^RANK_IC=" | grep -oE '[0-9]+\.[0-9]+' | head -1)
                    echo "${rank_ic}"


                    # score the portfolio and extract metrics
                    score_output=$(python score_submission.py "${submission}" \
                        --start "${test_start}" \
                        --end "${test_end}" 2>&1)

                    echo "${score_output}"

                    # parse the three return lines
                    port_ret=$(echo "${score_output}" | grep "portfolio return" | grep -oE '[+-][0-9]+\.[0-9]+' | head -1)
                    bench_ret=$(echo "${score_output}" | grep "benchmark return" | grep -oE '[+-][0-9]+\.[0-9]+' | head -1)
                    excess_ret=$(echo "${score_output}" | grep "excess return" | grep -oE '[+-][0-9]+\.[0-9]+' | head -1)

                    # append to summary CSV
                    echo "${ss},${cs},${mcw},${k},${label},${as_of},${test_start},${test_end},${port_ret},${bench_ret},${excess_ret},${rank_ic}" >> "${SUMMARY}"

                    # clean up temp submission
                    rm -f "${submission}"
                done
            done
        done
    done
done

echo ""
echo "=== Summary written to ${SUMMARY} ==="
cat "${SUMMARY}"