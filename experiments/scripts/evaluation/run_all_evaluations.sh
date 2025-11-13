#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_good_bad_plant_effect.py"

SAMPLED_DATA_ROOT="${PROJECT_DIR}/sampled_data"
DATA_ROOT="${PROJECT_DIR}/data"

REVIEW_ROOTS=("${SAMPLED_DATA_ROOT}" "${DATA_ROOT}")

run_eval() {
  local reviews_dir="$1"
  shift
  echo
  echo ">>> python ${EVAL_SCRIPT} --reviews_dir ${reviews_dir} $*"
  python "${EVAL_SCRIPT}" --reviews_dir "${reviews_dir}" "$@"
}

for root in "${REVIEW_ROOTS[@]}"; do
  [[ -d "${root}" ]] || continue
  
  for reviews_root in "${root}"/reviews_*; do
    [[ -d "${reviews_root}" ]] || continue
    
    for dataset_dir in "${reviews_root}"/*; do
      [[ -d "${dataset_dir}" ]] || continue
      
      case "$(basename "${dataset_dir}")" in
        evaluation_results|combined_visualizations) continue ;;
      esac
      
      if [[ -d "${dataset_dir}/planted_error" && -d "${dataset_dir}/sham_surgery" ]]; then
        run_eval "${dataset_dir}" --new_format
      fi
      
      if [[ -d "${dataset_dir}/latest" && -d "${dataset_dir}/authors_affiliation_good" && -d "${dataset_dir}/authors_affiliation_bad" ]]; then
        run_eval "${dataset_dir}" --pattern authors_affiliation --baseline latest
      fi
      
      if [[ -d "${dataset_dir}/latest" && -d "${dataset_dir}/v1" ]]; then
        run_eval "${dataset_dir}" --folders latest v1 latest
      fi
    done
  done
done

