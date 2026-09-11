import datetime as _dt
import itertools as _it
import json
from textwrap import indent

from pff import project_dir

# ────────────────────────────────────────────────────────────────
# 1) FIXED SETTINGS
# ────────────────────────────────────────────────────────────────
NEW_TAGS = ["AICME", "UAI-Feb"]
GPU_TYPE = "tesla_v100"  # or a100_40gb, tesla_v100
EPOCHS = 200
BATCH_SIZE = 64
TRAIN_SIZE = 12800
TEST_SIZE = 64
VAL_SIZE = 256
GPU_NUMBER = 1

PRETRAINING_TIME = 0.8
VAL_EVERY_PCT = 0.1
EMPIRICAL_EVERY_PCT = 0.1
MODEL_TYPES = ["aicme"]  # "snode", "nodepk", "cvae", "flow-pk"
DEBUG_TEST = False

NUM_WORKERS = 8

CONFIG_PATH_MAP = {
    "snode": str(
        project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    ),
    "nodepk": str(
        project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    ),
    "cvae": str(
        project_dir / "config_files" / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    ),
    "aicme": str(
        project_dir / "config_files" / "experiment_configs" / "UAI" / "aicme-t-pk" / "base.yaml"
    ),
    "flow-pk": str(
        project_dir
        / "config_files"
        / "experiment_configs"
        / "UAI"
        / "flow-pk-predict-n-generate"
        / "flowPK.yaml"
    ),
}

HF_MODEL_NAME_MAP = {
    "snode": "AICMEPK_cluster",
    "nodepk": "AICMEPK_cluster",
    "cvae": "AICMEPK_cluster",
    "aicme": "AICMEPK_cluster",
    "flow-pk": "AICMEPK_cluster",
}

PROTOCOL_MAP = {
    "snode": ["none"],
    "nodepk": ["none"],
    "cvae": ["none"],
    "aicme": ["none"],
    "flow-pk": ["none"],
}

# ────────────────────────────────────────────────────────────────
# 2) GRID SEARCH (override args)
# ────────────────────────────────────────────────────────────────
base_grid = {
    "INDIVIDUAL_ENCODER_NAME": ["RNNContextEncoder"],  # encoder stuff "RNNContextEncoderDosing"
    "TIME_OBS_ENCODER_HIDDEN_DIM": [256],
    "TIME_OBS_ENCODER_OUTPUT_DIM": [256],
    "OUTPUT_HEAD_NUMBER_LAYERS": [3],
    "RNN_INDIVIDUAL_ENCODER_NUMBER_OF_LAYERS": [4],  # [2,4]
    "ENCODER_RNN_HIDDEN_DIM": [256],
    "DECODER_NAME": [
        "TransformerDecoder",
        "RNNDecoder",
    ],  # decoder stuff RNNDecoder TransformerDecoder
    "INIT_HIDDEN_NUM_LAYERS": [2],  # [2,4]
    "DECODER_RNN_HIDDEN_DIM": [256],
    "DECODER_HIDDEN_DIM": [512],
    "RNN_DECODER_NUMBER_OF_LAYERS": [4],
    "ZI_LATENT_DIM": [256],
    "STUDY_LATENT_DETERMINISTIC": [False],
    "PREDICTION_LATENT_DETERMINISTIC": [False],
    "PREDICTION_ONLY": [True, False],
    "USE_KL_S": [True],
    "USE_KL_I": [True],
    "USE_KL_I_NP": [True],
    "USE_INVARIANCE_LOSS": [False],
    "USE_SELF_ATTENTION": [True],
    "USE_TIME_DELTAS": [True],
    "DECODER_NUM_LAYERS": [4],
    "LEARNING_RATE": [1e-4],
    "EMPIRICAL_NUMBER_OF_OBS": [False],
    "NODE_STEP": [True],
    "EXCLUSIVE_NODE_STEP": [True],
    "PRETRAINING_EPOCHS": [int(EPOCHS * PRETRAINING_TIME)],
    "AGGREGATOR_TYPE": ["mean"],
    "AGGREGATOR_NUM_HEADS": [8],
    "KEEP_TEMPFILE": [False],
    "STORE_IN_TEMPFILE": [False],
    "RECREATE_TEMPFILE": [False],
    "DOSING_SAME_ROUTE": [True],
    "LOSS_NAME": ["log_nll"],
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
tags_repr = (
    json.dumps(
        [*NEW_TAGS, "$TAG"],  # the list you want to pass
        separators=(",", ":"),  # remove “comma-space”
    ).replace('"', r"\"")  # turn " into \"
)


def bash_array(name, seq):
    inner = " ".join(f'"{x}"' for x in seq)
    return f"{name}=({inner})"


arrays_bash = "\n".join(bash_array(k, v) for k, v in columns.items())

# Template string
template = f"""#!/bin/bash
###############################################################################
# model-dynamic-array.job – auto-generated {_dt.datetime.now().isoformat(timespec="seconds")}
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

# ─── fixed hyper-params ─────────────────────────────────────────
EPOCHS={EPOCHS}
TRAIN_SIZE={TRAIN_SIZE}
BATCH_SIZE={BATCH_SIZE}
NUM_WORKERS={NUM_WORKERS}
TEST_SIZE={TEST_SIZE}
VAL_SIZE={VAL_SIZE}
VAL_EVERY_PCT={VAL_EVERY_PCT}
EMPIRICAL_EVERY_PCT={EMPIRICAL_EVERY_PCT}

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
REPO_ROOT="/home/ojedamarin/Projects/Pharma/pff"

# ─── override string ───────────────────────────────────────────
MODEL_TYPE_CURRENT=${{MODEL_TYPE[$RUN_INDEX]}}
if [[ "$MODEL_TYPE_CURRENT" == "flow-pk" ]]; then
  OVERRIDE_ARGS=""
else
  OVERRIDE=(
    train.epochs=$EPOCHS
    train.batch_size=$BATCH_SIZE
    train.num_workers=$NUM_WORKERS
    mix_data.train_size=$TRAIN_SIZE
    mix_data.test_size=$TEST_SIZE
    mix_data.val_size=$VAL_SIZE
    network.individual_encoder_name=${{INDIVIDUAL_ENCODER_NAME[$RUN_INDEX]}}
    network.time_obs_encoder_hidden_dim=${{TIME_OBS_ENCODER_HIDDEN_DIM[$RUN_INDEX]}}
    network.time_obs_encoder_output_dim=${{TIME_OBS_ENCODER_OUTPUT_DIM[$RUN_INDEX]}}
    network.rnn_individual_encoder_number_of_layers=${{RNN_INDIVIDUAL_ENCODER_NUMBER_OF_LAYERS[$RUN_INDEX]}}
    network.init_hidden_num_layers=${{INIT_HIDDEN_NUM_LAYERS[$RUN_INDEX]}}
    network.encoder_rnn_hidden_dim=${{ENCODER_RNN_HIDDEN_DIM[$RUN_INDEX]}}
    network.decoder_rnn_hidden_dim=${{DECODER_RNN_HIDDEN_DIM[$RUN_INDEX]}}
    network.decoder_hidden_dim=${{DECODER_HIDDEN_DIM[$RUN_INDEX]}}
    network.zi_latent_dim=${{ZI_LATENT_DIM[$RUN_INDEX]}}
    network.decoder_name=${{DECODER_NAME[$RUN_INDEX]}}
    network.decoder_num_layers=${{DECODER_NUM_LAYERS[$RUN_INDEX]}}
    network.rnn_decoder_number_of_layers=${{RNN_DECODER_NUMBER_OF_LAYERS[$RUN_INDEX]}}
    network.output_head_num_layers=${{OUTPUT_HEAD_NUMBER_LAYERS[$RUN_INDEX]}}
    network.exclusive_node_step=${{EXCLUSIVE_NODE_STEP[$RUN_INDEX]}}
    network.aggregator_type=${{AGGREGATOR_TYPE[$RUN_INDEX]}}
    network.aggregator_num_heads=${{AGGREGATOR_NUM_HEADS[$RUN_INDEX]}}
    network.use_self_attention=${{USE_SELF_ATTENTION[$RUN_INDEX]}}
    network.use_time_deltas=${{USE_TIME_DELTAS[$RUN_INDEX]}}
    network.use_invariance_loss=${{USE_INVARIANCE_LOSS[$RUN_INDEX]}}
    network.use_kl_i_np=${{USE_KL_I_NP[$RUN_INDEX]}}
    network.use_kl_i=${{USE_KL_I[$RUN_INDEX]}}
    network.use_kl_s=${{USE_KL_S[$RUN_INDEX]}}
    network.loss_name=${{LOSS_NAME[$RUN_INDEX]}}
    network.prediction_only=${{PREDICTION_ONLY[$RUN_INDEX]}}
    network.study_latent_deterministic=${{STUDY_LATENT_DETERMINISTIC[$RUN_INDEX]}}
    network.prediction_latent_deterministic=${{PREDICTION_LATENT_DETERMINISTIC[$RUN_INDEX]}}
    train.learning_rate=${{LEARNING_RATE[$RUN_INDEX]}}
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

  FILTERED=()
  for entry in "${{OVERRIDE[@]}}"; do
    if [[ ! "$entry" =~ None|none|NaN|nan ]]; then
      FILTERED+=("$entry")
    fi
  done

  OVERRIDE_ARGS="--override ${{FILTERED[*]}}"
fi


# ─── run ────────────────────────────────────────────────────────
echo "▶ Task $SLURM_ARRAY_TASK_ID | config=$(basename $CONFIG_PATH) | protocol=${{PRETRAINING_PROTOCOL[$RUN_INDEX]}} | tag=$TAG"
srun "$PYTHON_PATH" "$SCRIPT_PATH" --config_path "$CONFIG_PATH" $OVERRIDE_ARGS
"""

# Save the template
out_dir = project_dir / "scripts" / "slurms_jobs" / "Potsdam"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model-dynamic-array.job"
out_path.write_text(template)

print(f"✓ Wrote {out_path}  (total runs: {L})\n")
print(f"➡ To launch the sweep, run:\n   sbatch --array=0-{L - 1} {out_path.name}\n")
