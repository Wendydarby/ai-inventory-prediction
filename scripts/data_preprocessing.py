import pandas as pd

def clean_data(data):
    """
    Cleans the inventory dataset by handling missing values, removing duplicates,
    and ensuring correct data types.

    Args:
        data (str): Path to the CSV file containing the inventory data.

    Returns:
        pd.DataFrame: Cleaned inventory dataset.
    """

    print("\n--- CLEANING DATA ---")

    # Load dataset
    try:
        print(f"Loading dataset from: {data}")
        df = pd.read_csv(data)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {data}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        raise

    print(f"Rows loaded: {len(df)}")
    print("Columns:", list(df.columns))

    # Required columns check
    required = [
        "date", "sales_quantity", "price", "promotional_activity",
        "weather_data", "stock_level", "lead_time"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        print(f"[ERROR] Missing required columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")

    # Handle missing values
    print("Filling missing values...")
    try:
        df.fillna({
            "sales_quantity": 0,
            "price": df["price"].mean(),
            "promotional_activity": False,
            "weather_data": df["weather_data"].mean(),
            "stock_level": df["stock_level"].median(),
            "lead_time": df["lead_time"].median()
        }, inplace=True)
    except Exception as e:
        print(f"[ERROR] Failed to fill missing values: {e}")
        raise

    # Remove duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Removing duplicates: {duplicates} rows removed")
    df.drop_duplicates(inplace=True)

    # Data type conversions
    print("Converting data types...")
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["sales_quantity"] = df["sales_quantity"].astype(int)
        df["price"] = df["price"].astype(float)
        df["promotional_activity"] = df["promotional_activity"].astype(bool)
        df["stock_level"] = df["stock_level"].astype(int)
        df["lead_time"] = df["lead_time"].astype(int)
    except Exception as e:
        print(f"[ERROR] Failed to convert dtypes: {e}")
        raise

    # Range checks
    print("Applying range checks...")
    mask = (
        (df["sales_quantity"] >= 0) &
        (df["price"] >= 0) &
        (df["stock_level"] >= 0) &
        (df["lead_time"] >= 0)
    )
    removed = (~mask).sum()
    if removed > 0:
        print(f"Removing {removed} rows with invalid negative values")
    df = df[mask]

    # Formatting
    df["price"] = df["price"].round(2)

    # Summary
    print("\n--- CLEANING COMPLETE ---")
    print(f"Final row count: {len(df)}")
    print("--------------------------\n")

    return df
