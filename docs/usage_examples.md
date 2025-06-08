# Usage Examples

## 1. Loading and Cleaning Data

```python
from src.data_preprocessing import load_data, clean_data

df = load_data("../data/sample_inventory_data.csv")
df_clean = clean_data(df)