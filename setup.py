from setuptools import setup, find_packages

install_requires = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "lightning==2.4.0",
    "pandas==2.2.3",
    "torchdiffeq==0.2.5",
    "matplotlib==3.10.1",
    "comet_ml==3.49.6",
    "torchsde==0.2.6",
    "pytest==8.3.5",
    "gpytorch==1.14",
    "torchtyping==0.1.5",
    "transformers==4.52.4",
    "huggingface_hub==0.32.6",
    "datasets==4.1.1",
    "ruamel.yaml==0.19.0",
    "pot==0.9.6",
]

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pff",
    version="0.1.0",
    description="Prior-Fitted Functional Flows for pharmacokinetic modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="César Ojeda",
    author_email="cesarali07@gmail.com",
    packages=find_packages(include=["pff", "pff.*"]),
    install_requires=install_requires,
    python_requires=">=3.11.9",
    zip_safe=False,
)
