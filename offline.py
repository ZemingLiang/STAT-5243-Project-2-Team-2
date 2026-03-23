# offline_test.py

import io
import pandas as pd

from dataset_store import load_uploaded_csv, list_datasets
from feature_engineering import (
    apply_feature_engineering,
    apply_feature_engineering_to_df,
    feature_engineering_capabilities,
)

print("=== Test 1: dataframe-level feature engineering ===")
df = pd.DataFrame({
    "x": [1, 2, 3, 4],
    "y": [10, 20, 30, 40],
    "cat": ["A", "B", "A", "C"]
})

result1 = apply_feature_engineering_to_df(df, method="square", col1="x")
print(result1)

print("\n=== Test 2: one-hot encoding ===")
result2 = apply_feature_engineering_to_df(df, method="one_hot", col1="cat")
print(result2)

print("\n=== Test 3: dataset_store + feature_engineering integration ===")
csv_text = """x,y,cat
1,10,A
2,20,B
3,30,A
4,40,C
"""
file_obj = io.BytesIO(csv_text.encode("utf-8"))

meta = load_uploaded_csv(file_obj, filename="test_data.csv")
dataset_id = meta["dataset_id"]

print("Uploaded dataset_id:", dataset_id)
print("All datasets:", list_datasets())

result3 = apply_feature_engineering(
    dataset_id=dataset_id,
    method="interaction",
    col1="x",
    col2="y",
    save_mode="preview_only"
)
print(result3)

print("\n=== Test 4: overwrite mode ===")
result4 = apply_feature_engineering(
    dataset_id=dataset_id,
    method="log",
    col1="x",
    save_mode="overwrite"
)
print(result4)

print("\n=== Test 5: derived mode ===")
result5 = apply_feature_engineering(
    dataset_id=dataset_id,
    method="ratio",
    col1="y",
    col2="x",
    save_mode="derived"
)
print(result5)

print("\n=== Test 6: capabilities ===")
print(feature_engineering_capabilities())

print("\n=== Test 7: standardize ===")
result7 = apply_feature_engineering_to_df(df, method="standardize", col1="x")
print(result7)

print("\n=== Test 8: normalize ===")
result8 = apply_feature_engineering_to_df(df, method="normalize", col1="y")
print(result8)

df_missing = pd.DataFrame({
    "x": [1, None, 3, 4],
    "cat": ["A", None, "A", "C"]
})

print("\n=== Test 9: fillna mean ===")
result9 = apply_feature_engineering_to_df(df_missing, method="fillna", col1="x", strategy="mean")
print(result9)

print("\n=== Test 10: fillna mode ===")
result10 = apply_feature_engineering_to_df(df_missing, method="fillna", col1="cat", strategy="mode")
print(result10)

print("\n=== Test 11: dropna by column ===")
result11 = apply_feature_engineering_to_df(df_missing, method="dropna", col1="x")
print(result11)

print("\n=== Test 12: dropna all columns ===")
result12 = apply_feature_engineering_to_df(df_missing, method="dropna")
print(result12)

print("\nAll tests finished.")