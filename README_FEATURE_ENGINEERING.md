# Feature Engineering Module

## 1. Overview

The `feature_engineering.py` module provides backend utilities for transforming raw datasets into model-ready features. It is designed to work with both in-memory `pandas.DataFrame` objects and datasets managed by the backend storage layer through `dataset_id`.

This module is part of the project’s data-processing pipeline and is responsible for:
- creating new features from existing variables,
- applying common feature transformations,
- handling categorical encoding,
- supporting scaling and missing-value processing,
- allowing users to preview, overwrite, or save transformed datasets as derived datasets.

The module follows a JSON-friendly response style so that it can be integrated easily with frontend components or API endpoints.

---

## 2. Module Objectives

The feature engineering module is designed to satisfy the following goals:

1. Create new features that enhance data insights.
2. Provide an interactive way for users to generate and modify features.
3. Allow users to see the impact of transformation modifications through previews and summaries.
4. Support seamless integration with backend dataset storage.
5. Provide an extensible architecture for adding more feature methods later.

---

## 3. Supported Feature Engineering Methods

The module currently supports the following methods.

### 3.1 Log Transformation
- **Method name:** `log`
- **Formula:** `log_x = log1p(x)`
- **Input:** one numeric column
- **Output:** one numeric column
- **Validation:** all non-null values must be greater than or equal to `-1`

### 3.2 Square Transformation
- **Method name:** `square`
- **Formula:** `x_squared = x^2`
- **Input:** one numeric column
- **Output:** one numeric column

### 3.3 Cube Transformation
- **Method name:** `cube`
- **Formula:** `x_cubed = x^3`
- **Input:** one numeric column
- **Output:** one numeric column

### 3.4 Interaction Feature
- **Method name:** `interaction`
- **Formula:** `x_y = x * y`
- **Input:** two numeric columns
- **Output:** one numeric column

### 3.5 Ratio Feature
- **Method name:** `ratio`
- **Formula:** `y_div_x = y / x`
- **Input:** two numeric columns
- **Output:** one numeric column
- **Special behavior:** zero denominators are converted to `NaN` rather than causing a crash

### 3.6 Binning
- **Method name:** `binning`
- **Purpose:** discretize a numeric variable into bins
- **Input:** one numeric column
- **Output:** either integer-coded bins or interval labels
- **Options:** number of bins, labels, output column name

### 3.7 One-Hot Encoding
- **Method name:** `one_hot`
- **Purpose:** convert a categorical variable into multiple dummy variables
- **Input:** one categorical column
- **Output:** multiple numeric/boolean columns
- **Options:** prefix, `drop_first`

### 3.8 Standardization
- **Method name:** `standardize`
- **Formula:** `z = (x - mean) / std`
- **Input:** one numeric column
- **Output:** one numeric column
- **Use case:** scale numeric variables to comparable units

### 3.9 Normalization
- **Method name:** `normalize`
- **Formula:** `(x - min) / (max - min)`
- **Input:** one numeric column
- **Output:** one numeric column
- **Use case:** rescale values into the range `[0, 1]`

### 3.10 Fill Missing Values
- **Method name:** `fillna`
- **Purpose:** impute missing values in a column
- **Input:** one column
- **Supported strategies:**
  - `mean`
  - `median`
  - `mode`
  - `constant`
- **Options:** `fill_value`, optional output column name

### 3.11 Drop Missing Values
- **Method name:** `dropna`
- **Purpose:** remove rows containing missing values
- **Input:** optional target column
- **Behavior:**
  - if a column is provided, rows with missing values in that column are removed
  - if no column is provided, rows with missing values in any column are removed

---

## 4. Validation and Type Checking

The module includes several built-in validation rules to improve robustness and usability.

### 4.1 Column Validation
Before any transformation is applied, the module checks whether the required columns exist in the input dataset.

### 4.2 Numeric Validation
Methods such as `log`, `square`, `cube`, `interaction`, `ratio`, `binning`, `standardize`, and `normalize` require numeric input columns.

### 4.3 Categorical Validation
The `one_hot` method checks whether the target column is categorical-like before applying encoding.

### 4.4 Parameter Validation
Examples:
- `bins` must be at least 2 for binning
- `fill_value` must be provided when `strategy="constant"`
- `log1p` only works when values are at least `-1`

These checks prevent invalid operations and return meaningful error messages.

---

## 5. Core Functions

### 5.1 `apply_feature_engineering_to_df(df, method, ...)`

Applies feature engineering directly to a `pandas.DataFrame`.

#### Purpose
This function is useful for:
- local testing,
- backend development,
- internal reuse,
- validating transformations before storing results.

#### Main Inputs
- `df`: input DataFrame
- `method`: feature engineering method
- `col1`: primary input column
- `col2`: optional second input column
- optional method-specific parameters such as:
  - `bins`
  - `new_column`
  - `labels`
  - `prefix`
  - `drop_first`
  - `strategy`
  - `fill_value`

#### Output
Returns a standardized dictionary containing:
- status
- feature metadata
- dataset summary
- transformed preview
- message

### 5.2 `apply_feature_engineering(dataset_id, method, ...)`

Applies feature engineering to a dataset stored in the backend dataset store.

#### Workflow
1. Load dataset using `dataset_id`
2. Validate required columns and parameters
3. Apply the selected transformation
4. Return preview or save the result depending on the selected `save_mode`

#### Main Use Case
This function is intended for backend/API/frontend integration, where users interact with datasets through the storage system rather than raw DataFrames.

### 5.3 `feature_engineering_capabilities()`

Returns a JSON-friendly summary of all supported methods, including:
- method names
- labels
- required inputs
- configurable options
- output types
- supported save modes

#### Purpose
This is especially useful for frontend dropdown menus, UI configuration, or dynamically generated user guides.

---

## 6. Save Modes

The module supports three save strategies.

### 6.1 `preview_only`
- does not modify the dataset store
- returns only a preview of the transformed data
- useful for testing and interactive exploration

### 6.2 `overwrite`
- updates the original dataset in place
- keeps the same `dataset_id`
- useful when the user wants to permanently modify the stored dataset

### 6.3 `derived`
- creates a new dataset in the dataset store
- assigns a new `dataset_id`
- preserves the original dataset
- useful when the user wants to compare original and transformed versions

---

## 7. Integration with `dataset_store`

This module depends on the `dataset_store.py` module for backend data management.

### Functions used from `dataset_store`
- `get_dataset_by_id()`
- `get_dataset_metadata()`
- `update_dataset_by_id()`
- `create_derived_dataset()`

### Integration Workflow
- retrieve stored dataset,
- apply transformation,
- save result back to store if requested,
- return metadata and preview to the caller.

Because of this design, the feature engineering module does **not** manage file storage itself. It focuses only on transformation logic, while dataset persistence is delegated to the storage layer.

---

## 8. Example Workflows

### 8.1 DataFrame-Level Example

~~~python
import pandas as pd
from feature_engineering import apply_feature_engineering_to_df

df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [10, 20, 30, 40],
    "cat": ["A", "B", "A", "C"]
})

result = apply_feature_engineering_to_df(
    df=df,
    method="square",
    col1="x"
)

print(result)
~~~

### 8.2 Dataset Store Example

~~~python
from feature_engineering import apply_feature_engineering

result = apply_feature_engineering(
    dataset_id="some_dataset_id",
    method="log",
    col1="x",
    save_mode="overwrite"
)

print(result)
~~~

---

## 9. Standard Output Structure

All functions return a consistent JSON-style response.

Success
status: "success"
data:
feature_meta
new_columns
dataset_summary
preview
message: success message
Error
status: "error"
message: error description
details: exception info

---

## 10. Testing

The module was validated using an offline test script.

### 10.1 Covered tests

- dataframe-level transformations
- one-hot encoding
- dataset store integration
- preview mode
- overwrite mode
- derived mode
- capabilities endpoint
- scaling functions
- missing value handling

These tests confirm that the pipeline works both standalone and with backend integration.

---

## 11. Design Principles

This module follows several key design principles.

### 11.1 Modularity
transformation logic is separated into helper functions
easier to maintain and extend
### 11.2 Backend / Storage Separation
feature logic in feature_engineering.py
storage handled by dataset_store.py
### 11.3 Extensibility
add new helper function
extend dispatcher
update capabilities response
### 11.4 Frontend Readiness
JSON-style outputs
easy integration with UI
### 11.5 Error Transparency
clear validation checks
structured error messages