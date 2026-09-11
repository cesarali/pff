# Generative PK Dashboard: Setup & Testing Guide

## 📁 Folder Structure
```
pff/
├── api_service/                   ← Optional FastAPI + React service
│   └── backend/
│       ├── Dockerfile
│       └── main.py                 FastAPI app entry point
├── pff/                 ← Core machine learning library
│   ├── __init__.py
│   ├── config_classes/            ← Dataclasses for YAML/JSON configuration
│   ├── data/                      ← Empirical + simulated data tooling
│   │   ├── data_empirical/        – JSON schema, stats & batch builders
│   │   ├── data_generation/       – Simulation & compartment models
│   │   ├── data_preprocessing/    – Raw CSV → tensor pipelines
│   │   ├── datasets/              – Lightning-ready datamodules
│   │   └── extra/                 – Vectorised kernels & helpers
│   ├── metrics/                   ← Evaluation utilities
│   ├── models/                    ← Lightning modules & architectures
│   ├── training/                  ← Training loops & experiment entry points
│   └── utils/                     ← Shared helpers & plotting utilities
├── notebooks/                     ← Exploratory analyses / demos
├── scripts/                       ← CLI helpers
└── tests/                         ← Unit and integration tests
```

## API_Service

The API Service is to be deployed in the University Cluster.

### 🔐 University Cluster Login: Step-by-Step
✅ Prerequisites (one-time setup)

You received a kubeconfig file from your IT team (e.g. kubeconfig.yaml).
The cluster uses OIDC authentication — so you need the kubelogin plugin.
You have access to a namespace like: sfb1294-ojedamarin.

## Database handling
All data obtained via the frontend is handled via the backend and stored or retrieved from the MongoDB database if needed.

## Code Guidelines

Use torchtyping to specify torch tensors shapes.
At each step during forward pass or whenever possible with torch modules, specify using comments the different shapes of the tensors. SHAPE AWARENESS IS CRUCIAL.

ALWAYS PROVIDE DOCUMENTATION. NEVER DELETE DOCUMENTATION THAT WAS ALREADY THERE.
