# design_slurm_meta.py -- generate SLURM script sweeping MetaStudyConfig options

from pathlib import Path
from textwrap import indent
import datetime as _dt
import itertools as _it
from pff import project_dir
import json

# ────────────────────────────────────────────────────────────────
# 1) FIXED SETTINGS
# ────────────────────────────────────────────────────────────────
NEW_TAGS = ["A-0"]
GPU_TYPE = "tesla_v100"  # or a100_40gb, tesla_v100
EPOCHS = 500
BATCH_SIZE=128
TRAIN_SIZE = 1000
TEST_SIZE=256
VAL_SIZE=256
GPU_NUMBER=1

PRETRAINING_TIME = 0.8
LOG_IMAGE_EVERY_EPOCH = 25
MODEL_TYPES = ["aicme"] # "snode", "nodepk", "cvae"]
DEBUG_TEST = False


NUM_WORKERS=8

CONFIG_PATH_MAP = {
    "snode": str(project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"),
    "nodepk": str(project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"),
    "cvae": str(project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"),
    "aicme": str(project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"),
}

HF_MODEL_NAME_MAP = {
    "snode": "AICMEPK_cluster",
    "nodepk": "AICMEPK_cluster",
    "cvae": "AICMEPK_cluster",
    "aicme": "AICMEPK_cluster",
}

PROTOCOL_MAP = {
    "snode": ["none"],
    "nodepk": ["none", "fine-tuning", "exclusive"],
    "cvae": ["none"],
    "aicme": ["none"],
}
# ────────────────────────────────────────────────────────────────
# 2) GRID SEARCH (override args)
# ────────────────────────────────────────────────────────────────
base_grid = {
    # Search over MetaStudyConfig fields instead of network params
    "DRUG_ID_OPTIONS": ["['Drug_A','Drug_B','Drug_C']"],
    "NUM_INDIVIDUALS_RANGE": ["(6,10)"],
    "NUM_PERIPHERALS_RANGE": ["(1,3)"],
    "LOG_K_A_MEAN_RANGE": ["(-1,2)"],
    "LOG_K_A_STD_RANGE": ["(0.2,0.6)"],
    "K_A_TMAG_RANGE": ["(0.01,0.2)"],
    "K_A_TSCL_RANGE": ["(1,5)"],
    "LOG_K_E_MEAN_RANGE": ["(-5,0)"],
    "LOG_K_E_STD_RANGE": ["(0.2,0.6)"],
    "K_E_TMAG_RANGE": ["(0.01,0.2)"],
    "K_E_TSCL_RANGE": ["(1,5)"],
    "LOG_V_MEAN_RANGE": ["(2,8)"],
    "LOG_V_STD_RANGE": ["(0.2,0.6)"],
    "V_TMAG_RANGE": ["(0.001,0.0099)"],
    "V_TSCL_RANGE": ["(1,5)"],
    "LOG_K_1P_MEAN_RANGE": ["(-4,0)"],
    "LOG_K_1P_STD_RANGE": ["(0.2,0.6)"],
    "K_1P_TMAG_RANGE": ["(0.01,0.2)"],
    "K_1P_TSCL_RANGE": ["(1,5)"],
    "LOG_K_P1_MEAN_RANGE": ["(-4,-1)"],
    "LOG_K_P1_STD_RANGE": ["(0.2,0.6)"],
    "K_P1_TMAG_RANGE": ["(0.01,0.2)"],
    "K_P1_TSCL_RANGE": ["(1,5)"],
    "REL_RUV_RANGE": ["(0.05,0.3)"],
    "TIME_START": [0.0],
    "TIME_STOP": [10.0],
    "TIME_NUM_STEPS": [100],
    "SOLVER_METHOD": ["rk4"],
    "USE_SELF_ATTENTION": [True],
    "USE_TIME_DELTAS": [True],
    "EMPIRICAL_NUMBER_OF_OBS": [False],
    "KEEP_TEMPFILE": [False],
    "STORE_IN_TEMPFILE": [False],
    "RECREATE_TEMPFILE": [False],
    "DOSING_SAME_ROUTE": [True, False],
    "LEARNING_RATE": [1e-4],
}

combos = []
base_keys, base_vals = zip(*base_grid.items())
for base_combo in _it.product(*base_vals):
    base_dict = dict(zip(base_keys, base_combo))
    for mtype in MODEL_TYPES:
        for protocol in PROTOCOL_MAP[mtype]:
            combo = base_dict.copy()
            combo["MODEL_TYPE"] = mtype
            combo["CONFIG_PATH"] = CONFIG_PATH_MAP[mtype]
            combo["PRETRAINING_PROTOCOL"] = protocol
            combo["HF_MODEL_NAME"] = HF_MODEL_NAME_MAP[mtype]
            combos.append(combo)

keys = list(combos[0].keys())
columns = {k: [c[k] for c in combos] for k in keys}

L = len(combos)
tags_repr = json.dumps(
    [*NEW_TAGS, "$TAG"],       # the list you want to pass
    separators=(",", ":")                     # remove “comma-space”
).replace('"', r'\"')                         # turn " into \"

def bash_array(name, seq):
    # Quote each item so bash treats items containing parentheses or
    # commas as literal strings rather than attempting arithmetic.
    inner = " ".join(f'"{x}"' for x in seq)
    return f"{name}=({inner})"

arrays_bash = "\n".join(bash_array(k, v) for k, v in columns.items())

# Template string (we will print it in the next cell to avoid truncation)
template = f"""#!/bin/bash
###############################################################################
# model-dynamic-array.job – auto-generated { _dt.datetime.now().isoformat(timespec="seconds") }
###############################################################################
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus={GPU_TYPE}:{GPU_NUMBER}
#SBATCH --ntasks-per-node={GPU_NUMBER}
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=0-18:00
#SBATCH --chdir=/work/ojedamarin/testjobs
#SBATCH --mail-type=ALL
#SBATCH --output=/work/ojedamarin/testjobs/slurm-%A_%a.out
#SBATCH --array=0-{L-1}
###############################################################################

# proxies & env
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128
export ftp_proxy=http://proxy2.uni-potsdam.de:3128

module load lang/Miniforge3/24.1.2-0


# ─── fixed hyper-params ─────────────────────────────────────────
EPOCHS={EPOCHS}
LOG_IMAGE_EVERY_EPOCH={LOG_IMAGE_EVERY_EPOCH}
TRAIN_SIZE={TRAIN_SIZE}
BATCH_SIZE={BATCH_SIZE}
TEST_SIZE={TEST_SIZE}
VAL_SIZE={TRAIN_SIZE}
NUM_WORKERS=8

# ─── auto-generated parameter arrays ───────────────────────────
{indent(arrays_bash, "")}

# ─── array index ───────────────────────────────────────────────
RUN_INDEX=$SLURM_ARRAY_TASK_ID

# ─── tag logic ─────────────────────────────────────────────────
CONFIG_PATH=${{CONFIG_PATH[$RUN_INDEX]}}

if [[ "$CONFIG_PATH" == *"snode-pk"* ]]; then
  TAG="S-0"
elif [[ "$CONFIG_PATH" == *"cvae-node-pk"* ]]; then
  TAG="C-0"
elif [[ "$CONFIG_PATH" == *"neural-process"* ]]; then
  TAG="N-0"
else
  case "${{PRETRAINING_PROTOCOL[$RUN_INDEX]}}" in
    none)        TAG="B-0" ;;
    exclusive)   TAG="B-1" ;;
    fine-tuning) TAG="B-2" ;;
    *)           TAG="debug" ;;
  esac
fi


# ─── script paths ──────────────────────────────────────────────
PYTHON_PATH="/home/ojedamarin/.conda/envs/sim-priors-pk/bin/python"
SCRIPT_PATH="/home/ojedamarin/Projects/Pharma/pff/scripts/training/train_model.py"
MY_RESULTS_PATH="/work/ojedamarin/Projects/Pharma/Results/"

# ─── override string ───────────────────────────────────────────
OVERRIDE=(
  train.epochs=$EPOCHS
  train.batch_size=$BATCH_SIZE
  train.num_workers=$NUM_WORKERS
  train.log_image_every_epoch=$LOG_IMAGE_EVERY_EPOCH
  mix_data.train_size=$TRAIN_SIZE
  mix_data.test_size=$TEST_SIZE
  mix_data.val_size=$VAL_SIZE
  network.use_self_attention=${{USE_SELF_ATTENTION[$RUN_INDEX]}}
  network.use_time_deltas=${{USE_TIME_DELTAS[$RUN_INDEX]}}
  train.learning_rate=${{LEARNING_RATE[$RUN_INDEX]}}
  meta_study.drug_id_options=${{DRUG_ID_OPTIONS[$RUN_INDEX]}}
  meta_study.num_individuals_range=${{NUM_INDIVIDUALS_RANGE[$RUN_INDEX]}}
  meta_study.num_peripherals_range=${{NUM_PERIPHERALS_RANGE[$RUN_INDEX]}}
  meta_study.log_k_a_mean_range=${{LOG_K_A_MEAN_RANGE[$RUN_INDEX]}}
  meta_study.log_k_a_std_range=${{LOG_K_A_STD_RANGE[$RUN_INDEX]}}
  meta_study.k_a_tmag_range=${{K_A_TMAG_RANGE[$RUN_INDEX]}}
  meta_study.k_a_tscl_range=${{K_A_TSCL_RANGE[$RUN_INDEX]}}
  meta_study.log_k_e_mean_range=${{LOG_K_E_MEAN_RANGE[$RUN_INDEX]}}
  meta_study.log_k_e_std_range=${{LOG_K_E_STD_RANGE[$RUN_INDEX]}}
  meta_study.k_e_tmag_range=${{K_E_TMAG_RANGE[$RUN_INDEX]}}
  meta_study.k_e_tscl_range=${{K_E_TSCL_RANGE[$RUN_INDEX]}}
  meta_study.log_V_mean_range=${{LOG_V_MEAN_RANGE[$RUN_INDEX]}}
  meta_study.log_V_std_range=${{LOG_V_STD_RANGE[$RUN_INDEX]}}
  meta_study.V_tmag_range=${{V_TMAG_RANGE[$RUN_INDEX]}}
  meta_study.V_tscl_range=${{V_TSCL_RANGE[$RUN_INDEX]}}
  meta_study.log_k_1p_mean_range=${{LOG_K_1P_MEAN_RANGE[$RUN_INDEX]}}
  meta_study.log_k_1p_std_range=${{LOG_K_1P_STD_RANGE[$RUN_INDEX]}}
  meta_study.k_1p_tmag_range=${{K_1P_TMAG_RANGE[$RUN_INDEX]}}
  meta_study.k_1p_tscl_range=${{K_1P_TSCL_RANGE[$RUN_INDEX]}}
  meta_study.log_k_p1_mean_range=${{LOG_K_P1_MEAN_RANGE[$RUN_INDEX]}}
  meta_study.log_k_p1_std_range=${{LOG_K_P1_STD_RANGE[$RUN_INDEX]}}
  meta_study.k_p1_tmag_range=${{K_P1_TMAG_RANGE[$RUN_INDEX]}}
  meta_study.k_p1_tscl_range=${{K_P1_TSCL_RANGE[$RUN_INDEX]}}
  meta_study.rel_ruv_range=${{REL_RUV_RANGE[$RUN_INDEX]}}
  meta_study.time_start=${{TIME_START[$RUN_INDEX]}}
  meta_study.time_stop=${{TIME_STOP[$RUN_INDEX]}}
  meta_study.time_num_steps=${{TIME_NUM_STEPS[$RUN_INDEX]}}
  meta_study.solver_method=${{SOLVER_METHOD[$RUN_INDEX]}}
  mix_data.pretraining_protocol=${{PRETRAINING_PROTOCOL[$RUN_INDEX]}}
  mix_data.pretraining_epochs=${{PRETRAINING_EPOCHS[$RUN_INDEX]}}
  mix_data.keep_tempfile=${{KEEP_TEMPFILE[$RUN_INDEX]}}
  mix_data.store_in_tempfile=${{STORE_IN_TEMPFILE[$RUN_INDEX]}}
  mix_data.recreate_tempfile=${{RECREATE_TEMPFILE[$RUN_INDEX]}}
  dosing.same_route=${{DOSING_SAME_ROUTE[$RUN_INDEX]}}
  context_observations.empirical_number_of_obs=${{EMPIRICAL_NUMBER_OF_OBS[$RUN_INDEX]}}
  "tags={tags_repr}"
  run_index=$RUN_INDEX
  my_results_path=$MY_RESULTS_PATH
  hf_model_name=${{HF_MODEL_NAME[$RUN_INDEX]}}
  debug_test=$DEBUG_TEST
)
OVERRIDE_ARGS="--override ${{OVERRIDE[*]}}"

# ─── run ────────────────────────────────────────────────────────
echo "▶ Task $SLURM_ARRAY_TASK_ID | config=$(basename $CONFIG_PATH) | protocol=${{PRETRAINING_PROTOCOL[$RUN_INDEX]}} | tag=$TAG"
$PYTHON_PATH $SCRIPT_PATH --config_path $CONFIG_PATH $OVERRIDE_ARGS
"""

# Save the template to simulate the output
out_dir = project_dir / "scripts" / "slurms_jobs"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model-dynamic-array-meta.job"
out_path.write_text(template)


print(f"✓ Wrote {out_path}  (total runs: {L})\n")
print(f"➡ To launch the sweep, run:\n   sbatch --array=0-{L-1} {out_path.name}\n")
