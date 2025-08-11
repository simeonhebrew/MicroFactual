# Current Architecture - Microbiome ML Package

## System Context Diagram (C4 Level 1)

```mermaid
C4Context
    title System Context - Microbiome ML Package

    Person(scientist, "Microbiome Researcher", "Scientists studying microbiome data who need ML analysis tools")
    Person(advanced_user, "Advanced ML User", "Users who want to customize model parameters and workflows")
    
    System(microbiome_ml, "Microbiome ML Package", "Python package for microbiome data analysis using machine learning")
    
    SystemDb(input_data, "Input Data Files", "Abundance tables and metadata files (TSV/CSV format)")
    SystemDb(output_data, "Output Results", "Predictions, visualizations, and model artifacts")
    
    Rel(scientist, microbiome_ml, "Uses via CLI", "Command line interface")
    Rel(advanced_user, microbiome_ml, "Uses via notebooks", "Jupyter notebooks")
    Rel(microbiome_ml, input_data, "Reads", "File I/O")
    Rel(microbiome_ml, output_data, "Writes", "File I/O")
```

## Container Diagram (C4 Level 2)

```mermaid
C4Container
    title Container Diagram - Microbiome ML Package

    Person(user, "User", "Microbiome researcher")
    
    Container_Boundary(microbiome_ml_boundary, "Microbiome ML Package") {
        Container(cli_interface, "CLI Interface", "Python/argparse", "Command line entry point for the pipeline")
        Container(pipeline_orchestrator, "Pipeline Orchestrator", "Python", "Coordinates the ML workflow execution")
        Container(data_processor, "Data Processing Module", "Python/pandas/numpy", "Loads, filters, and transforms microbiome data")
        Container(ml_engine, "ML Engine", "Python/scikit-learn", "Trains and evaluates machine learning models")
        Container(visualization_engine, "Visualization Engine", "Python/matplotlib", "Generates plots and saves results")
        Container(utilities, "Utilities", "Python", "Logging, configuration, and helper functions")
    }
    
    SystemDb(file_system, "File System", "Input data files and output results")
    
    Rel(user, cli_interface, "Executes commands", "CLI")
    Rel(cli_interface, pipeline_orchestrator, "Calls", "Function calls")
    Rel(pipeline_orchestrator, data_processor, "Uses", "Data loading & transformation")
    Rel(pipeline_orchestrator, ml_engine, "Uses", "Model training & prediction")
    Rel(pipeline_orchestrator, visualization_engine, "Uses", "Result visualization")
    Rel(pipeline_orchestrator, utilities, "Uses", "Logging & utilities")
    
    Rel(data_processor, file_system, "Reads", "File I/O")
    Rel(visualization_engine, file_system, "Writes", "File I/O")
```

## Component Diagram (C4 Level 3) - Current Implementation

```mermaid
C4Component
    title Component Diagram - Current Implementation

    Container_Boundary(microbiome_ml_boundary, "Microbiome ML Package") {
        Component(main_py, "main.py", "Python", "Entry point with run_pipeline() and main() functions")
        
        Component(data_processing_py, "data_processing.py", "Python", "load_data(), filter_data(), clr_transform()")
        Component(modeling_py, "modeling.py", "Python", "train_model() with RandomForest + GridSearchCV")
        Component(visualization_py, "visualization.py", "Python", "plot_roc(), save_roc_curve(), save_probabilities()")
        Component(utils_py, "utils.py", "Python", "get_logger(), parse_args(), save_results()")
        
        Component(constants, "Constants", "Python", "File names and configuration values")
    }
    
    SystemDb(input_files, "Input Files", "abundance_*.txt, metadata_*.txt")
    SystemDb(output_files, "Output Files", "predicted_probabilities.csv, roc_curve.png")
    
    Rel(main_py, data_processing_py, "Uses", "Data pipeline")
    Rel(main_py, modeling_py, "Uses", "ML training")
    Rel(main_py, visualization_py, "Uses", "Result saving")
    Rel(main_py, utils_py, "Uses", "Utilities")
    
    Rel(data_processing_py, input_files, "Reads", "File I/O")
    Rel(visualization_py, output_files, "Writes", "File I/O")
```

## Data Flow Diagram

```mermaid
flowchart TD
    A[Input Files<br/>abundance.txt + metadata.txt] --> B[load_data]
    B --> C[filter_data<br/>abundance & prevalence]
    C --> D[clr_transform<br/>centered log-ratio]
    D --> E[train_model<br/>RandomForest + GridSearchCV]
    E --> F[predict_proba<br/>generate predictions]
    F --> G{Output Directory<br/>specified?}
    G -->|Yes| H[save_results]
    G -->|No| I[Pipeline Complete]
    H --> J[create_output_dir]
    H --> K[save_probabilities<br/>CSV file]
    H --> L[save_roc_curve<br/>PNG plot]
    J --> I
    K --> I
    L --> I
```
