# Hub Runtime Bundle

This directory contains the parallel Hugging Face export path for
consumer-facing model bundles.

The existing training export remains unchanged:

- native export: `BasicLightningExperiment._push_model_to_hub(...)`
- runtime export: `push_loaded_model_runtime_bundle(...)`

The runtime export is intended for users who should be able to load a model
from the Hugging Face Hub through `transformers` without installing the local
`pff` package.

## Important Constraint

The consumer entrypoint is `transformers`, but `transformers` alone is **not**
enough today.

These runtime bundles still execute PyTorch-based custom code and reconstruct
the internal PK architecture, so the user needs the runtime Python
dependencies, but not a local checkout of this repository.

Reliable consumer install:

```bash
pip install torch transformers huggingface_hub lightning datasets pandas torchtyping gpytorch pot torchdiffeq torchsde ruamel.yaml pyyaml
```

What the consumer does **not** need:

- `pip install pff`
- a local clone of this repository
- access to the training checkpoint directory

## Consumer Workflow

Use the runtime repo, not the native training-artifact repo.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/your-model-runtime",
    trust_remote_code=True,
)
```

Then call the stable runtime task API:

```python
outputs = model.run_task(
    task="generate",   # or "predict"
    studies=studies,   # one StudyJSON or a list[StudyJSON]
    num_samples=8,
)
```

The return payload is:

```python
{
    "task": "generate",
    "io_schema_version": "studyjson-v1",
    "model_info": {...},
    "results": [
        {
            "input_index": 0,
            "samples": [study_json_0, study_json_1, ...],
        }
    ],
}
```

## Generate Example

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/your-model-runtime",
    trust_remote_code=True,
)

studies = [
    {
        "context": [
            {
                "name_id": "ctx_0",
                "observations": [0.2, 0.5, 0.3],
                "observation_times": [0.5, 1.0, 2.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }
        ],
        "target": [],
        "meta_data": {
            "study_name": "demo",
            "substance_name": "drug_x",
        },
    }
]

outputs = model.run_task(
    task="generate",
    studies=studies,
    num_samples=4,
)

generated_studies = outputs["results"][0]["samples"]
```

## Predict Example

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "your-org/your-model-runtime",
    trust_remote_code=True,
)

predict_studies = [
    {
        "context": [
            {
                "name_id": "ctx_0",
                "observations": [0.2, 0.5, 0.3],
                "observation_times": [0.5, 1.0, 2.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }
        ],
        "target": [
            {
                "name_id": "tgt_0",
                "observations": [0.25, 0.31],
                "observation_times": [0.5, 1.0],
                "remaining": [0.0, 0.0, 0.0],
                "remaining_times": [2.0, 4.0, 8.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }
        ],
        "meta_data": {
            "study_name": "demo",
            "substance_name": "drug_x",
        },
    }
]

outputs = model.run_task(
    task="predict",
    studies=predict_studies,
    num_samples=4,
)

prediction_samples = outputs["results"][0]["samples"]
```

## Producer Workflow

To publish a runtime repo from a locally loaded experiment:

```python
from pff.hub_runtime import push_loaded_model_runtime_bundle

runtime_repo_id = push_loaded_model_runtime_bundle(
    experiment=experiment,
    model_card_path=["hf_model_cards", "AICME-PK_Readme.md"],
)
```

By default this creates a separate repo:

```text
<namespace>/<hf_model_name>-runtime
```

That keeps the native training artifact export and the consumer runtime export
separate.
