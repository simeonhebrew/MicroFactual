# Makefile for ML_Microbiome_Package

.PHONY : venv activate install run test clean help
# Variables
ENV_NAME ?= ml_microbiome_env
PYTHON_VERSION ?= python3.9
ABUNDANCE_FILE ?= Dataset/abundance_crc.txt
METADATA_FILE ?= Dataset/metadata_crc.txt
TARGET_COLUMN ?= Group
OUTPUT_DIR ?= output

# Create a virtual environment
venv:
	uv venv $(ENV_NAME) --python $(PYTHON_VERSION)

# Activate the virtual environment
activate:
	@echo "Run 'source $(ENV_NAME)/bin/activate' to activate the virtual environment."

# Install dependencies and the package
install:
	uv pip install -e .

# Run the package
run:
	uv run microbiome-ml --abundance $(ABUNDANCE_FILE) --metadata $(METADATA_FILE) --target $(TARGET_COLUMN) --output_dir $(OUTPUT_DIR)

test:
	uv run pytest
# Clean the output directory
clean:
	rm -rf $(OUTPUT_DIR)

# Help
help:
	@echo "Makefile for ML_Microbiome_Package"
	@echo "Targets:"
	@echo "  venv       - Create a virtual environment"
	@echo "  activate   - Instructions to activate the virtual environment"
	@echo "  install    - Install dependencies and the package"
	@echo "  run        - Run the package with default or specified parameters"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean the output directory"