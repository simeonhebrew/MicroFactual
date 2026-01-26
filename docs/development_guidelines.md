# MicroML Development Guidelines

## 🎯 Core Development Principles

### **1. Code Philosophy**
- **Concise code as much as possible** - Favor readability over cleverness
- **Simple is better** - Follow the Zen of Python: "Simple is better than complex"
- **Domain-driven design** - Code should reflect microbiome domain concepts
- **API consistency** - Maintain consistent patterns across all components

### **2. Architecture Adherence**
- **Always refer to the overall architecture** - Every component should fit the proposed layered design
- **Discuss architectural conflicts** - If implementation conflicts with architecture, bring it up for discussion before proceeding
- **Maintain backward compatibility** during transitions when possible
- **Document architectural decisions** in code comments and ADRs (Architecture Decision Records)

### **3. Testing Strategy**
- **Test-driven development (TDD)** where possible - Write tests first, then implementation
- **Robust and concise tests** - Tests should be comprehensive but not verbose
- **Domain-specific test data** - Use realistic microbiome datasets in tests
- **Property-based testing** for data transformations

### **4. Type Safety & Documentation**
- **Type hints everywhere** - All functions, methods, and variables should have type annotations
- **Docstrings follow numpy style** - Consistent with scientific Python ecosystem
- **Type checking in CI** - mypy integration in GitHub Actions
- **Runtime type validation** for public APIs using pydantic or similar

## 📋 Detailed Guidelines

### **Code Quality Standards**

#### **Naming Conventions**
```python
# Classes: PascalCase with domain context
class MicrobiomeDataset:
class CLRTransform:
class RandomForestClassifier:

# Functions/methods: snake_case, descriptive verbs
def load_abundance_data():
def filter_by_prevalence():
def train_classification_model():

# Variables: snake_case, descriptive nouns
abundance_matrix: pd.DataFrame
sample_metadata: pd.DataFrame
feature_importance_scores: np.ndarray

# Constants: UPPER_SNAKE_CASE
DEFAULT_PREVALENCE_THRESHOLD = 0.05
MIN_ABUNDANCE_CUTOFF = 1e-6
```

#### **Function Design**
```python
# Good: Single responsibility, clear interface
def filter_by_abundance(
    data: pd.DataFrame,
    threshold: float = 1e-6
) -> pd.DataFrame:
    """Filter features by minimum abundance threshold."""
    return data[data.mean(axis=1) >= threshold]

# Avoid: Multiple responsibilities, unclear parameters
def process_data(data, params):  # Too vague
```

#### **Error Handling**
```python
# Use domain-specific exceptions
class MicrobiomeDataError(Exception):
    """Base exception for microbiome data issues."""

class InsufficientSamplesError(MicrobiomeDataError):
    """Raised when dataset has too few samples for analysis."""

# Validate inputs early and clearly
def clr_transform(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise MicrobiomeDataError("Cannot transform empty dataset")
    if (data < 0).any().any():
        raise MicrobiomeDataError("Abundance data cannot contain negative values")
    # ... transform logic
```

### **Testing Guidelines**

#### **Test Structure**
```python
# Use pytest with clear test organization
class TestCLRTransform:
    """Test suite for CLR transformation."""

    def test_clr_transform_basic_functionality(self):
        """Test CLR transform produces expected output."""
        # Arrange
        data = create_test_abundance_data()
        expected = calculate_expected_clr(data)

        # Act
        result = clr_transform(data)

        # Assert
        np.testing.assert_array_almost_equal(result, expected)

    def test_clr_transform_handles_zeros(self):
        """Test CLR transform handles zero values correctly."""
        # Test specific edge cases

    @pytest.mark.parametrize("n_samples,n_features", [(10, 50), (100, 200)])
    def test_clr_transform_different_sizes(self, n_samples, n_features):
        """Test CLR transform works with different data sizes."""
        # Property-based testing
```

#### **Test Data Management**
```python
# Use fixtures for consistent test data
@pytest.fixture
def sample_abundance_data():
    """Create realistic microbiome abundance data for testing."""
    return pd.DataFrame({
        'sample_1': [0.1, 0.2, 0.0, 0.7],
        'sample_2': [0.05, 0.15, 0.3, 0.5],
        # ... more samples
    }, index=['species_A', 'species_B', 'species_C', 'species_D'])

@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return pd.DataFrame({
        'sample_id': ['sample_1', 'sample_2'],
        'condition': ['healthy', 'disease'],
        'age': [25, 30]
    })
```

### **Documentation Standards**

#### **Docstring Format**
```python
def train_microbiome_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "random_forest",
    cv_folds: int = 5,
    **kwargs
) -> Tuple[BaseEstimator, Dict[str, float]]:
    """Train a microbiome classification model with cross-validation.

    This function trains a machine learning model on microbiome abundance
    data and evaluates it using stratified cross-validation to ensure
    robust performance estimates.

    Parameters
    ----------
    X : pd.DataFrame
        Abundance matrix with samples as rows and features as columns.
        Should be preprocessed (filtered, normalized, transformed).
    y : pd.Series
        Target labels for classification. Index should match X.
    model_type : str, default="random_forest"
        Type of classifier to train. Options: {"random_forest", "svm", "logistic"}.
    cv_folds : int, default=5
        Number of cross-validation folds for evaluation.
    **kwargs
        Additional parameters passed to the model constructor.

    Returns
    -------
    model : BaseEstimator
        Trained scikit-learn compatible model.
    scores : Dict[str, float]
        Cross-validation scores including accuracy, precision, recall, f1.

    Raises
    ------
    MicrobiomeDataError
        If X and y have mismatched indices or insufficient samples.
    ValueError
        If model_type is not supported.

    Examples
    --------
    >>> abundance = load_abundance_data("data.tsv")
    >>> labels = load_labels("metadata.tsv")
    >>> model, scores = train_microbiome_classifier(abundance, labels)
    >>> print(f"Cross-validation accuracy: {scores['accuracy']:.3f}")

    Notes
    -----
    This function assumes the input data has been appropriately preprocessed
    for microbiome analysis (e.g., filtered for prevalence and abundance,
    transformed using CLR or similar compositional data transformation).

    See Also
    --------
    preprocess_microbiome_data : Recommended preprocessing pipeline
    evaluate_model_performance : Detailed model evaluation utilities
    """
```

### **Architecture Compliance**

#### **Component Design Patterns**
```python
# Follow the established base classes
class MicrobiomeRandomForest(MicrobiomeClassifier):
    """Random Forest classifier optimized for microbiome data."""

    def __init__(
        self,
        n_estimators: int = 100,
        preprocessing: Optional[Union[str, Pipeline]] = "auto",
        feature_selection: Optional[str] = None,
        **sklearn_params
    ):
        super().__init__(preprocessing=preprocessing)
        self.n_estimators = n_estimators
        self.feature_selection = feature_selection
        self.sklearn_params = sklearn_params

    def _create_base_estimator(self) -> RandomForestClassifier:
        """Create the underlying sklearn estimator."""
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            **self.sklearn_params
        )
```

#### **Data Structure Consistency**
```python
# Always work with MicrobiomeDataset when possible
def analyze_differential_abundance(
    dataset: MicrobiomeDataset,
    comparison_column: str,
    method: str = "deseq2"
) -> pd.DataFrame:
    """Analyze differential abundance between groups."""
    # Access components through the dataset interface
    abundance = dataset.abundance_matrix
    metadata = dataset.sample_metadata

    # Validate that comparison column exists
    if comparison_column not in metadata.columns:
        raise MicrobiomeDataError(
            f"Comparison column '{comparison_column}' not found in metadata"
        )

    # ... analysis logic
```

### **Performance & Scalability**

#### **Memory Efficiency**
```python
# Use generators for large datasets
def process_samples_in_batches(
    dataset: MicrobiomeDataset,
    batch_size: int = 1000
) -> Iterator[pd.DataFrame]:
    """Process samples in batches to manage memory usage."""
    for i in range(0, len(dataset), batch_size):
        yield dataset.abundance_matrix.iloc[i:i+batch_size]

# Lazy loading for large files
class LazyMicrobiomeDataset(MicrobiomeDataset):
    """Dataset that loads data on demand."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._abundance_matrix = None

    @property
    def abundance_matrix(self) -> pd.DataFrame:
        if self._abundance_matrix is None:
            self._abundance_matrix = pd.read_csv(self.file_path, sep='\t')
        return self._abundance_matrix
```

#### **Computational Efficiency**
```python
# Use numpy operations when possible
def calculate_clr_transform_vectorized(data: np.ndarray) -> np.ndarray:
    """Vectorized CLR transformation for better performance."""
    # Replace loops with numpy operations
    log_data = np.log(data + 1e-6)  # Add pseudocount
    geometric_means = np.exp(np.mean(log_data, axis=1, keepdims=True))
    return log_data - np.log(geometric_means)

# Parallel processing for expensive operations
from joblib import Parallel, delayed

def cross_validate_parallel(
    model, X, y, cv_folds: int = 5, n_jobs: int = -1
) -> List[float]:
    """Parallel cross-validation for faster evaluation."""
    return Parallel(n_jobs=n_jobs)(
        delayed(single_fold_validation)(model, X, y, fold)
        for fold in range(cv_folds)
    )
```

### **Version Control & Collaboration**

#### **Commit Message Standards**
```
# Format: type(scope): description

feat(core): add MicrobiomeDataset class with metadata support
fix(preprocessing): handle zero values in CLR transformation
docs(api): update docstrings for classification module
test(models): add property-based tests for feature selection
refactor(utils): simplify logging configuration
chore(ci): update GitHub Actions workflow for Python 3.12
```

#### **Branch Naming**
```
# Feature branches
feature/microbiome-dataset-implementation
feature/counterfactual-explanations
feature/ensemble-methods

# Bug fixes
fix/clr-transform-zero-handling
fix/metadata-loading-encoding-issue

# Architecture changes
arch/plugin-system-design
arch/pipeline-refactor
```

#### **Pull Request Guidelines**
- **Reference architecture** - Explain how changes fit the overall design
- **Include tests** - All new functionality must have tests
- **Update documentation** - Code changes should include doc updates
- **Performance considerations** - Note any performance implications
- **Breaking changes** - Clearly mark and justify any API changes

### **Continuous Integration Requirements**

#### **Pre-commit Hooks**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

#### **CI Pipeline Requirements**
- **All tests pass** across Python 3.10, 3.11, 3.12
- **Type checking** with mypy passes
- **Code formatting** with ruff format
- **Test coverage** ≥ 90% for core modules
- **Documentation builds** successfully
- **No security vulnerabilities** (safety check)

### **Release Process**

#### **Semantic Versioning**
- **MAJOR** (1.0.0): Architecture changes, breaking API changes
- **MINOR** (0.1.0): New features, backwards-compatible additions
- **PATCH** (0.0.1): Bug fixes, documentation updates

#### **Changelog Maintenance**
- Keep CHANGELOG.md updated with each PR
- Categorize changes: Added, Changed, Deprecated, Removed, Fixed, Security
- Link to relevant issues and PRs

---

## 🚀 Implementation Checklist

When implementing new features:

- [ ] **Design Review**: Does this fit the architecture?
- [ ] **API Design**: Is the interface intuitive and consistent?
- [ ] **Write Tests**: TDD approach, comprehensive coverage
- [ ] **Type Hints**: Full type annotation
- [ ] **Documentation**: Numpy-style docstrings + examples
- [ ] **Performance**: Consider memory and computational efficiency
- [ ] **Error Handling**: Appropriate exceptions and validation
- [ ] **Integration**: Works with existing components
- [ ] **Review**: Code review focusing on architecture alignment

## 📚 Additional Resources

- [Scientific Python Development Guide](https://learn.scientific-python.org/development/)
- [Scikit-learn Contributor Guide](https://scikit-learn.org/stable/developers/index.html)
- [Effective Python by Brett Slatkin](https://effectivepython.com/)
- [Clean Architecture by Robert Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

---

*These guidelines are living documents - they should evolve as we learn and the project grows.*
