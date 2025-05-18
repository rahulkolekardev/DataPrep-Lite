
<p align="center">
  <!-- Optional: Add a logo here if you have one -->
  <!-- <img src="path/to/your/logo.png" alt="DataPrep-Lite Logo" width="200"/> -->
  <h1 align="center">DataPrep-Lite</h1>
</p>

<p align="center">
  A lightweight and user-friendly Python library for common data cleaning and preprocessing tasks.
</p>

<p align="center">

  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License: MIT">
  </a>

</p>

---

**DataPrep-Lite** simplifies your data preparation workflows by providing a suite of intuitive, configurable, and pipeline-able transformers, designed to work seamlessly with Pandas DataFrames and inspired by the scikit-learn API.

## ✨ Features

*   **🐼 Pandas-Centric:** Natively operates on Pandas DataFrames.
*   **🤖 Scikit-learn Inspired API:** Familiar `fit`, `transform`, and `fit_transform` methods.
*   **🧱 Modular & Reusable:** Each operation is a distinct, configurable transformer.
*   **🔗 Pipeline Ready:** Easily chain multiple cleaning and preprocessing steps using the `Pipeline` class.
*   **⚙️ Configurable:** Transformers offer parameters to customize their behavior for various scenarios.
*   **🚀 Lightweight:** Focuses on common, essential tasks without excessive dependencies.

## 📚 Table of Contents

*   [Installation](#installation)
*   [Quick Start](#quick-start)
*   [Available Transformers](#available-transformers)
    *   [Core Components](#core-components)
    *   [Cleaning Transformers](#cleaning-transformers)
    *   [Preprocessing Transformers](#preprocessing-transformers)
*   [Usage Examples](#usage-examples)
    *   [Basic Pipeline](#basic-pipeline)
    *   [Using Individual Transformers](#using-individual-transformers)
    *   [Custom Column Selection](#custom-column-selection)
*   [Contributing](#contributing)
*   [Roadmap](#roadmap)
*   [License](#license)

## 🛠 Installation

### From PyPI (Recommended)

Once DataPrep-Lite is published to PyPI, you can install it using pip:

```bash
pip install dataprep-lite
```

### From Source

To install the latest development version directly from the GitHub repository:

```bash
git clone https://github.com/rahulkolekardev/DataPrep-Lite.git
cd DataPrep-Lite
pip install .
```

For development (editable install):

```bash
pip install -e .
# If you have development dependencies defined in setup.py or pyproject.toml:
# pip install -e ".[dev]"
```

**Dependencies:**

*   Python 3.8+
*   Pandas (>=1.3.0)
*   NumPy (>=1.20.0)
*   Scikit-learn (>=1.0.0)

## 🚀 Quick Start

Get up and running with DataPrep-Lite in minutes:

```python
import pandas as pd
import numpy as np

# Import necessary components from dataprep_lite
from dataprep_lite.core import Pipeline
from dataprep_lite.cleaning import MeanImputer, DropDuplicates, TypeConverter
from dataprep_lite.preprocessing import OneHotEncoderWrapper, MinMaxScalerWrapper

# Sample DataFrame
data = {
    'age': [25, 30, np.nan, 22, 30, 35],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Tokyo'],
    'experience_years': ['5', '10', '3', '2', '10', '12.5'], # String type
    'salary': [70000, 90000, 65000, 50000, 90000, 110000.0]
}
df = pd.DataFrame(data)
print("Original DataFrame:\n", df)

# Define a preprocessing pipeline
pipeline = Pipeline([
    ('type_converter', TypeConverter(type_mapping={'experience_years': 'to_numeric'})),
    ('age_imputer', MeanImputer(columns_to_process=['age'])),
    ('duplicate_remover', DropDuplicates(subset=['age', 'city'], keep='first')),
    ('city_ohe', OneHotEncoderWrapper(columns_to_process=['city'], drop='first', sparse_output=False)),
    ('scaler', MinMaxScalerWrapper(columns_to_process=['age', 'experience_years', 'salary']))
])

# Apply the pipeline
df_cleaned = pipeline.fit_transform(df.copy()) # Use .copy() to preserve original

print("\nCleaned DataFrame:\n", df_cleaned.head())
print("\nCleaned DataFrame dtypes:\n", df_cleaned.dtypes)
```

## 🧩 Available Transformers

### Core Components (`dataprep_lite.core`)

*   **`BaseTransformer`**: The abstract base class for all transformers. Enables creation of custom transformers compatible with the library's pipeline.
*   **`Pipeline`**: Chains multiple transformer steps into a single workflow object.
*   **Utility Functions**: (e.g., `identify_numeric_columns`, `identify_categorical_columns`) for internal use and potentially for users.

### Cleaning Transformers (`dataprep_lite.cleaning`)

*   **Missing Value Handlers:**
    *   `MeanImputer(columns_to_process=None)`: Imputes with mean.
    *   `MedianImputer(columns_to_process=None)`: Imputes with median.
    *   `ModeImputer(columns_to_process=None)`: Imputes with mode.
    *   `ConstantImputer(fill_value, columns_to_process=None)`: Imputes with a constant.
    *   `DropMissing(axis=0, how='any', thresh=None, subset=None)`: Drops rows/columns with NaNs.
*   **`DropDuplicates(subset=None, keep='first', ignore_index=False)`**: Removes duplicate rows.
*   **`OutlierIQRHandler(columns_to_process=None, factor=1.5, action='cap')`**: Handles outliers using IQR (`action` can be `'cap'` or `'remove_rows'`).
*   **`TypeConverter(type_mapping, errors='raise')`**: Converts column data types (e.g., `{'col': 'int64'}`, `{'col': 'to_numeric'}`).
*   **`BasicTextCleaner(columns_to_process=None, lowercase=True, strip_whitespace=True, remove_punctuation=True, ...)`**: Performs basic text cleaning.

### Preprocessing Transformers (`dataprep_lite.preprocessing`)

*   **Encoding:**
    *   `OneHotEncoderWrapper(...)`: Wraps `sklearn.preprocessing.OneHotEncoder`.
    *   `LabelEncoderWrapper(...)`: Wraps `sklearn.preprocessing.LabelEncoder` (column-wise).
*   **Scaling:**
    *   `MinMaxScalerWrapper(...)`: Wraps `sklearn.preprocessing.MinMaxScaler`.
    *   `StandardScalerWrapper(...)`: Wraps `sklearn.preprocessing.StandardScaler`.
*   **`KBinsDiscretizerWrapper(...)`**: Bins continuous data. Wraps `sklearn.preprocessing.KBinsDiscretizer`.
*   **`DatetimeFeatureCreator(...)`**: Extracts features (year, month, day, hour, etc.) from datetime columns.

*For detailed parameters of each transformer, please refer to their respective docstrings or the source code.*

## 📋 Usage Examples

### Basic Pipeline

```python
from dataprep_lite.core import Pipeline
from dataprep_lite.cleaning import MedianImputer, DropDuplicates
from dataprep_lite.preprocessing import StandardScalerWrapper
# ... (assuming df is your DataFrame)

pipeline = Pipeline([
    ('imputer', MedianImputer(columns_to_process=['feature1', 'feature2'])),
    ('duplicates', DropDuplicates()),
    ('scaler', StandardScalerWrapper()) # Applies to all numeric columns by default
])

df_processed = pipeline.fit_transform(df.copy())
```

### Using Individual Transformers

```python
from dataprep_lite.cleaning import OutlierIQRHandler
# ... (assuming df is your DataFrame)

outlier_handler = OutlierIQRHandler(columns_to_process=['salary'], action='cap', factor=2.0)
df_no_outliers = outlier_handler.fit_transform(df.copy())
```

### Custom Column Selection

Most transformers accept a `columns_to_process` parameter:

```python
from dataprep_lite.cleaning import ModeImputer
# ... (assuming df is your DataFrame)

# Impute mode only for 'category_A' and 'category_B'
mode_imputer = ModeImputer(columns_to_process=['category_A', 'category_B'])
df_imputed = mode_imputer.fit_transform(df.copy())
```

If `columns_to_process` is not provided, transformers attempt to apply to suitable default columns (e.g., numeric columns for `MeanImputer`, text-like columns for `BasicTextCleaner`).

## 🙌 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  **Fork the Project** (from `https://github.com/rahulkolekardev/DataPrep-Lite`)
2.  **Create your Feature Branch** (`git checkout -b feature/AmazingFeature`)
3.  **Commit your Changes** (`git commit -m 'Add some AmazingFeature'`)
4.  **Push to the Branch** (`git push origin feature/AmazingFeature`)
5.  **Open a Pull Request** (against `rahulkolekardev/DataPrep-Lite:main`)

Please ensure your code adheres to common Python styling (e.g., PEP 8), and include tests for new features or bug fixes.

You can also contribute by:
*   Reporting bugs or issues on the [GitHub Issues page](https://github.com/rahulkolekardev/DataPrep-Lite/issues).
*   Suggesting new features or enhancements.
*   Improving documentation.

## 🗺️ Roadmap

*   [ ] More advanced imputation techniques (e.g., KNNImputer wrapper).
*   [ ] Additional text processing features (e.g., TF-IDF, stop word removal).
*   [ ] Feature selection transformers (e.g., variance threshold, correlation-based).
*   [ ] Enhanced reporting and logging capabilities for applied transformations.
*   [ ] More robust handling of mixed-type columns in various transformers.
*   [ ] Comprehensive Sphinx documentation hosted on ReadTheDocs.
*   [ ] More extensive examples and tutorials.
*   [ ] Integration with other data science ecosystem tools.

See the [open issues](https://github.com/rahulkolekardev/DataPrep-Lite/issues) for a full list of proposed features (and known issues).

## 📜 License

Distributed under the MIT License. See `LICENSE` file for more information.
(You will need to create a `LICENSE` file in your repository, typically with the MIT License text).

```text
MIT License

Copyright (c) [Year] [Your Name or Organization - e.g., Rahul Kolekar]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgements

*   Inspired by the ease of use and design patterns of [scikit-learn](https://scikit-learn.org/).
*   Built upon the power of [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/).

---


