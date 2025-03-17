# CoDeSurv

This repository contains implementation for coCPM. For demonstration, it uses CoDeSurv and DeSurv models.

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CoDeSurv.git
cd CoDeSurv
```

2. Install the package using pip:
```bash
pip install .
```

Or install in development mode:
```bash
pip install -e .
```

### Dependencies
The main dependencies will be installed automatically through pip. Key dependencies include:
- PyTorch
- lifelines
- scikit-learn

## Project Structure

- `src/models/` - Contains the main model implementations
  - `CoDeSurv.py` - Implementation of the CoDeSurv model
  - `DeSurv.py` - Implementation of the DeSurv model
- `src/utils/` - Utility functions and helper modules
- `scripts/` - Additional scripts for data generation and processing
- `notebook/` - Jupyter notebook demonstration of the method

## Usage

Import and use the models:

```python
from src.models.coDeSurv import CoDeSurv
from src.models.DeSurv import DeSurv

# Initialize models
codesurv_model = CoDeSurv()
desurv_model = DeSurv()
```