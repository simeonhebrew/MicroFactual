# MicroFactual

MicroFactual is a user-friendly python framework for performing interpretable microbiome machine learning workflows based on counterfactual explanations.

## Recent Updates (June 2025)

- **Improved Flexibility**: Added support for configurable sample ID column names via the `--sample_column_name` parameter (default: "Sample ID")
- **Code Standardization**: Improved code formatting and consistency across the codebase

### Features to implement

- Allowing for the integration of linear and tree-based models

- Allowing for user parameter optimization at each workflow level

- Parallelization


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
uv run microfactual --abundance <abundance_file> --metadata <metadata_file> --target <target_column> --output_dir <output_directory> --sample_column_name <sample_id_column>
```
Where:
- `<abundance_file>`: Path to the abundance data file.
- `<metadata_file>`: Path to the metadata file.
- `<target_column>`: The target column for the model.
- `<output_directory>`: Path to the output directory where results will be saved.
- `<sample_id_column>`: (Optional) The column name for sample IDs in metadata (default: "Sample ID").

### Using the Makefile

You can use the provided `Makefile` to simplify common tasks in the project. Below are the available targets:

- **Create a virtual environment**:
  ```bash
  make venv
  ```
  This will create a virtual environment using `uv` with the specified Python version.

- **Activate the virtual environment**:
  ```bash
  make activate
  ```
  This will display instructions to activate the virtual environment manually.

- **Install dependencies and the package**:
  ```bash
  make install
  ```
  This will install all dependencies and the package itself.

- **Run the package**:
  ```bash
  make run
  ```
  This will execute the pipeline with default or specified parameters. You can customize the parameters by editing the `Makefile` or passing them as environment variables.

- **Run tests**:
  ```bash
  make test
  ```
  This will run the tests using `pytest`. Make sure to have the test files in the appropriate directory.

- **Clean the output directory**:
  ```bash
  make clean
  ```
  This will remove the output directory and its contents.

- **Display help**:
  ```bash
  make help
  ```
  This will display a list of available targets and their descriptions.
