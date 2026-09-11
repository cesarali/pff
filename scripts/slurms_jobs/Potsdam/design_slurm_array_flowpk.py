import datetime as _dt
import itertools as _it
import json
from textwrap import indent

from pff import project_dir

# ────────────────────────────────────────────────────────────────
# 1) FIXED SETTINGS
# ────────────────────────────────────────────────────────────────
NEW_TAGS = ["UAI", "FlowPK-Grid"]
GPU_TYPE = "tesla_v100"  # or a100_40gb, tesla_v100
GPU_NUMBER = 1

EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 8
TRAIN_SIZE = 12800
TEST_SIZE = 64
VAL_SIZE = 64
VAL_EVERY_PCT = 0.1
EMPIRICAL_EVERY_PCT = 0.1
DEBUG_TEST = False

CONFIG_PATH = str(
    project_dir
    / "config_files"
    / "experiment_configs"
    / "UAI"
    / "flow-pk-predict-n-generate"
    / "flowPK.yaml"
)
HF_MODEL_NAME = "FFlowPK_cluster"

# File names relative to CONFIG_PATH directory. This is only a selector,
# the file content is loaded by scripts/training/train_model.py.
META_STUDY_NAMES = [
    "base.meta_study.yaml",
    "literature-informed-heterogeneous.meta_study.yaml",
    # "literature-informed-homogeneous.meta_study.yaml",
]
# ────────────────────────────────────────────────────────────────
# 2) GRID SEARCH (FlowPK-only + meta-study selector)
# ────────────────────────────────────────────────────────────────
base_grid = {
    # source_process
    "SOURCE_TYPE": ["white_noise"],
    "FLOW_SIGMA": [1.0e-4],
    "GP_VARIANCE": [0.02, 0.1],
    "GP_LENGTH_SCALE": [0.04, 0.01],
    "GP_EPS": [0.001],
    "GP_TRANSFORM": [None],
    # vector_field
    "VF_HIDDEN_DIM": [256, 512],
    "VF_FOURIER_MODES": [20],
    "VF_USE_SPECTRAL_QKV": [False],
    "VF_TIME_FOURIER_MAX_FREQ": [128],
    "VF_ENCODER_NUM_HEADS": [4],
    "VF_DECODER_NUM_HEADS": [4],
    "VF_ENCODER_ATTN_LAYERS": [8],
    "VF_DECODER_ATTN_LAYERS": [8],
    "VF_DROPOUT": [0.2],
    # keep training settings in this script
    "LEARNING_RATE": [1.0e-4],
    # meta-study selection
    "META_STUDY_NAME": META_STUDY_NAMES,
}

combos = []
base_keys, base_vals = zip(*base_grid.items())
for base_combo in _it.product(*base_vals):
    combos.append(dict(zip(base_keys, base_combo)))

if not combos:
    raise RuntimeError("No FlowPK sweep combinations were generated.")

keys = list(combos[0].keys())
columns = {k: [c[k] for c in combos] for k in keys}

L = len(combos)
tags_repr = json.dumps([*NEW_TAGS, "$TAG"], separators=(",", ":")).replace('"', r"\"")


def bash_array(name, seq):
    inner = " ".join(f'"{x}"' for x in seq)
    return f"{name}=({inner})"


arrays_bash = "\n".join(bash_array(k, v) for k, v in columns.items())

# Template string
template = f"""#!/bin/bash
###############################################################################
# model-dynamic-array-flowpk.job – auto-generated {_dt.datetime.now().isoformat(timespec="seconds")}
###############################################################################
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus={GPU_TYPE}:{GPU_NUMBER}
#SBATCH --ntasks-per-node={GPU_NUMBER}
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=0-24:00
#SBATCH --chdir=/work/ojedamarin/pff_runs
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/work/ojedamarin/pff_runs/slurm-%A_%a.out
#SBATCH --array=0-{L - 1}
###############################################################################

# proxies & env
export http_proxy=http://proxy2.uni-potsdam.de:3128
export https_proxy=http://proxy2.uni-potsdam.de:3128
export ftp_proxy=http://proxy2.uni-potsdam.de:3128

# module load lang/Miniforge3/24.1.2-0
# conda activate sim-priors-pk

# ─── fixed run settings ─────────────────────────────────────────
EPOCHS={EPOCHS}
TRAIN_SIZE={TRAIN_SIZE}
BATCH_SIZE={BATCH_SIZE}
NUM_WORKERS={NUM_WORKERS}
TEST_SIZE={TEST_SIZE}
VAL_SIZE={VAL_SIZE}
VAL_EVERY_PCT={VAL_EVERY_PCT}
EMPIRICAL_EVERY_PCT={EMPIRICAL_EVERY_PCT}
CONFIG_PATH="{CONFIG_PATH}"

# ─── auto-generated parameter arrays ───────────────────────────
{indent(arrays_bash, "")}

# ─── array index ───────────────────────────────────────────────
RUN_INDEX=$SLURM_ARRAY_TASK_ID
TAG="FLOWPK"

# ─── script paths ──────────────────────────────────────────────
PYTHON_PATH="/home/ojedamarin/.conda/envs/sim-priors-pk/bin/python"
SCRIPT_PATH="/home/ojedamarin/Projects/Pharma/pff/scripts/training/train_model.py"
MY_RESULTS_PATH="/work/ojedamarin/Projects/Pharma/Results/"

# ─── override string ───────────────────────────────────────────
  OVERRIDE=(
    train.epochs=$EPOCHS
    train.batch_size=$BATCH_SIZE
    train.num_workers=$NUM_WORKERS
    train.learning_rate=${{LEARNING_RATE[$RUN_INDEX]}}

    mix_data.train_size=$TRAIN_SIZE
    mix_data.test_size=$TEST_SIZE
    mix_data.val_size=$VAL_SIZE

    source_process.source_type=${{SOURCE_TYPE[$RUN_INDEX]}}
    source_process.flow_sigma=${{FLOW_SIGMA[$RUN_INDEX]}}
    source_process.gp_variance=${{GP_VARIANCE[$RUN_INDEX]}}
    source_process.gp_length_scale=${{GP_LENGTH_SCALE[$RUN_INDEX]}}
    source_process.gp_eps=${{GP_EPS[$RUN_INDEX]}}
    source_process.gp_transform=${{GP_TRANSFORM[$RUN_INDEX]}}

    vector_field.hidden_dim=${{VF_HIDDEN_DIM[$RUN_INDEX]}}
    vector_field.fourier_modes=${{VF_FOURIER_MODES[$RUN_INDEX]}}
    vector_field.use_spectral_qkv=${{VF_USE_SPECTRAL_QKV[$RUN_INDEX]}}
    vector_field.time_fourier_max_freq=${{VF_TIME_FOURIER_MAX_FREQ[$RUN_INDEX]}}
    vector_field.encoder_num_heads=${{VF_ENCODER_NUM_HEADS[$RUN_INDEX]}}
    vector_field.decoder_num_heads=${{VF_DECODER_NUM_HEADS[$RUN_INDEX]}}
    vector_field.encoder_attention_layers=${{VF_ENCODER_ATTN_LAYERS[$RUN_INDEX]}}
    vector_field.decoder_attention_layers=${{VF_DECODER_ATTN_LAYERS[$RUN_INDEX]}}
    vector_field.dropout=${{VF_DROPOUT[$RUN_INDEX]}}

    "tags={tags_repr}"
    run_index=$RUN_INDEX
    my_results_path=$MY_RESULTS_PATH
    hf_model_name={HF_MODEL_NAME}
    debug_test={DEBUG_TEST}
)

FILTERED=()
for entry in "${{OVERRIDE[@]}}"; do
  if [[ ! "$entry" =~ None|none|NaN|nan ]]; then
    FILTERED+=("$entry")
  fi
done

OVERRIDE_ARGS="--override ${{FILTERED[*]}}"
META_STUDY_NAME_CURRENT=${{META_STUDY_NAME[$RUN_INDEX]}}

# ─── run ────────────────────────────────────────────────────────
echo "▶ Task $SLURM_ARRAY_TASK_ID | meta=$META_STUDY_NAME_CURRENT | tag=$TAG"
srun "$PYTHON_PATH" "$SCRIPT_PATH" \\
  --config_path "$CONFIG_PATH" \\
  --meta_study_name "$META_STUDY_NAME_CURRENT" \\
  $OVERRIDE_ARGS
"""

# Save the template
out_dir = project_dir / "scripts" / "slurms_jobs" / "Potsdam"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model-dynamic-array-flowpk.job"
out_path.write_text(template)

print(f"✓ Wrote {out_path}  (total runs: {L})\n")
print(f"➡ To launch the sweep, run:\n   sbatch --array=0-{L - 1} {out_path.name}\n")
