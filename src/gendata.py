import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker


def generate_inventory_data(num_records=10000, start_date=datetime(2023, 1, 1)):
    """
    Generates synthetic inventory data for testing prediction models.

    Args:
        num_records (int): The total number of records to generate.
        start_date (datetime): The start date for the historical data.

    Returns:
        pd.DataFrame: A DataFrame containing the generated data.
    """
    fake = Faker()
    end_date = start_date + timedelta(days=730)

    # Define product categories and names
    categories = ['Equipment', 'Apparel', 'Nutrition', 'Collectibles', 'Accessories']
    products_per_category = 10

    products = []
    for category in categories:
        for i in range(products_per_category):
            products.append({
                'product_id': f'PROD-{category[:3].upper()}-{i + 1:03d}',
                'category': category,
                'base_demand': np.random.randint(50, 150),
                'price_sensitivity': np.random.uniform(0.1, 0.5)
            })

    # Generate records
    data = []
    current_date = start_date
    while current_date <= end_date:
        for product in products:
            # Base sales quantity
            sales_quantity = product['base_demand'] + np.random.randint(-10, 10)

            # Seasonal trend (e.g., winter vs. summer)
            month = current_date.month
            if product['category'] == 'Apparel':
                if month in [12, 1, 2]:  # Winter wear
                    sales_quantity *= np.random.uniform(1.2, 1.5)
                elif month in [6, 7, 8]:  # Summer wear
                    sales_quantity *= np.random.uniform(0.8, 1.1)

            # Promotional impact
            is_promo = np.random.rand() < 0.1  # 10% chance of a promotion
            if is_promo:
                sales_quantity *= np.random.uniform(1.5, 2.5)  # Sales spike

            # Random fluctuations and market noise
            sales_quantity *= np.random.uniform(0.9, 1.1)

            # Price and promotional activity
            base_price = np.random.uniform(10.0, 500.0)
            price = base_price * (1 - is_promo * product['price_sensitivity'] * 0.5)  # Price drop for promos

            # Weather data (simplified)
            weather_data = np.random.uniform(15.0, 30.0) + 5 * np.sin(
                2 * np.pi * (current_date.timetuple().tm_yday / 365))

            # Stock levels and lead time
            stock_level = int(sales_quantity * np.random.uniform(1.5, 3.0))  # Stock is usually higher than sales
            lead_time = np.random.randint(1, 14)

            data.append({
                'product_id': product['product_id'],
                'date': current_date,
                'sales_quantity': max(0, int(sales_quantity)),
                'price': round(price, 2),
                'promotional_activity': is_promo,
                'weather_data': round(weather_data, 2),
                'stock_level': stock_level,
                'lead_time': lead_time
            })

            # Stop if we've reached the desired number of records
            if len(data) >= num_records:
                break

        current_date += timedelta(days=1)
        if len(data) >= num_records:
            break

    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    # Generate data
    df = generate_inventory_data(num_records=10000)

    # Save to CSV
    output_path = 'data/sample_inventory_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records and saved to {output_path}")

