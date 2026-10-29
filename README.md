# Can We Trust Feature Importance? Testing Explanations of Machine Learning Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Student Research Hackathon — Computer Science**  
*Improving the Interpretability and Reliability of Machine-Learning Models*

**Author:** Arpan, Diploma AIML  
**Email:** vikasakshat1977@gmail.com  
**GitHub:** [Astrino04](https://github.com/Astrino04)

---

## Abstract

When a model makes predictions, it is often useful to know what factors were most important. However, there are no guarantees that the importance scores are accurate. This project trains three models on the breast cancer dataset, evaluates two methods of determining feature importance, and performs three simple tests to determine if the results are meaningful:

1. **Agreement** — How consistent the models are with each other
2. **Stability** — Whether results remain consistent after retraining
3. **Removal** — Whether removing important features actually harms the model

While the models achieved similar accuracies (93.6%–95.9%), they were inconsistent with each other on feature importance. Permutation importance was unstable after retraining. However, the built-in importance of the random forest remained consistent and passed the removal test.

## Project Structure

```
feature-importance-trust/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py                    # Run the complete experiment
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Load and split the breast cancer dataset
│   ├── models.py              # Train the three ML models
│   ├── importance.py          # Compute feature importance (built-in & permutation)
│   ├── tests_agreement.py     # Test 1: Agreement between models
│   ├── tests_stability.py     # Test 2: Stability over retrainings
│   ├── tests_removal.py       # Test 3: Removal of top features
│   └── visualization.py       # Generate all plots
├── results/                   # CSV outputs (auto-generated)
├── plots/                     # PNG plots (auto-generated)
└── paper/
    └── paper.docx             # Original research paper
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Astrino04/feature-importance-trust.git
cd feature-importance-trust

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run the Complete Experiment

```bash
python main.py
```

This will:
- Load the Wisconsin Breast Cancer dataset
- Train Logistic Regression, Random Forest, and Gradient Boosting models
- Compute built-in and permutation feature importance
- Run all three tests (Agreement, Stability, Removal)
- Generate result tables (saved to `results/`)
- Generate visualizations (saved to `plots/`)

### Run Individual Components

```python
from src.data_loader import load_breast_cancer_data
from src.models import train_models
from src.importance import get_builtin_importance, get_permutation_importance

# Load data
X_train, X_test, y_train, y_test, feature_names = load_breast_cancer_data()

# Train models
models = train_models(X_train, y_train)

# Get importance scores
builtin_imp = get_builtin_importance(models, feature_names)
perm_imp = get_permutation_importance(models, X_test, y_test, feature_names)
```

## Key Results

### Model Accuracy
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 95.9% |
| Random Forest | 95.3% |
| Gradient Boosting | 93.6% |

### Stability (Average Correlation over 30 Retrainings)
| Model | Permutation Importance | Built-in Importance |
|-------|----------------------|-------------------|
| Logistic Regression | 0.41 | 0.61 |
| Random Forest | 0.20 | **0.92** |
| Gradient Boosting | 0.20 | 0.49 |

### Removal Test (Accuracy after Removing 10 Features)
| Model | Nothing Removed | Top 10 Built-in | Top 10 Permutation | 10 Random |
|-------|----------------|-----------------|-------------------|-----------|
| Logistic Regression | 95.9% | 95.3% | 90.1% | 94.8% |
| Random Forest | 95.3% | **73.7%** | 92.4% | 93.6% |
| Gradient Boosting | 93.6% | **69.0%** | 75.4% | 92.9% |

## References

1. M. T. Ribeiro, S. Singh, C. Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*, Proc. ACM SIGKDD, 2016.
2. S. M. Lundberg, S.-I. Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS, 2017.
3. L. Breiman, *Random Forests*, Machine Learning, 45 (2001) 5–32.
4. C. Strobl et al., *Conditional Variable Importance for Random Forests*, BMC Bioinformatics, 9 (2008) 307.
5. F. Doshi-Velez, B. Kim, *Towards a Rigorous Science of Interpretable Machine Learning*, arXiv:1702.08608, 2017.
6. F. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 12 (2011) 2825–2830.
7. W. H. Wolberg, W. N. Street, O. L. Mangasarian, *Breast Cancer Wisconsin (Diagnostic)*, UCI ML Repository, 1995.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
