# ML_Microbiome_Package

A user-friendly python framework for microbiome machine learning 
workflows.


### Features to implement

- Allowing for the use of different ML models

- Intergrating a layer of explainability of the ML models

- Allowing for user parameter optimization at each step

- Parallelization

- Workflow structure

## Executing python version

We use uv as the python package manager to create a virtual environment with the required dependencies.
To install uv, see the following link:
[https://docs.astral.sh/uv/#installation](https://docs.astral.sh/uv/#installation)

### Installing the package
To create a virtual environment with the required dependencies, run the following command in your terminal:
```bash
uv venv <env_name> --python <python_version>
```
To activate the virtual environment, run the following command in your terminal:
```bash
source <env_name>/bin/activate
```
Install the project to install all the dependencies as well as the package itself:
```bash
uv pip install -e .
```

### Running the package
To run the package, you can use the following command:
```bash
uv run main.py --abundance <abundance_file> --metadata <metadata_file> --target <target_column> --model <model_name> --output_dir <output_directory>
```
Where:
- `<abundance_file>`: Path to the abundance data file.
- `<metadata_file>`: Path to the metadata file.
- `<target_column>`: The target column for the model.
- `<model_name>`: The name of the machine learning model to use.
- `<output_directory>`: Path to the output directory where results will be saved.
```