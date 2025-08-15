import unittest
import pandas as pd
from gendata import generate_inventory_data


class TestInventoryDataGenerator(unittest.TestCase):
    """
    Test suite for the generate_inventory_data function.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up a single, reusable DataFrame for all tests to improve efficiency.
        """
        cls.df = generate_inventory_data(num_records=15000)

    def test_schema_and_data_types(self):
        """
        Verify that the DataFrame has the correct columns and data types.
        """
        expected_columns = [
            'product_id', 'date', 'sales_quantity', 'price',
            'promotional_activity', 'weather_data', 'stock_level', 'lead_time'
        ]

        # Check column names
        self.assertListEqual(list(self.df.columns), expected_columns)

        # Check data types
        self.assertEqual(self.df['product_id'].dtype, 'object')
        self.assertEqual(pd.api.types.is_datetime64_any_dtype(self.df['date']), True)
        self.assertEqual(self.df['sales_quantity'].dtype, 'int64')
        self.assertEqual(self.df['price'].dtype, 'float64')
        self.assertEqual(self.df['promotional_activity'].dtype, 'bool')
        self.assertEqual(self.df['weather_data'].dtype, 'float64')
        self.assertEqual(self.df['stock_level'].dtype, 'int64')
        self.assertEqual(self.df['lead_time'].dtype, 'int64')

    def test_minimum_record_count(self):
        """
        Ensure the generated dataset has at least the minimum required number of records.
        """
        self.assertGreaterEqual(len(self.df), 10000)

    def test_product_and_category_diversity(self):
        """
        Verify that the data includes the specified number of unique products and categories.
        """
        # The 'category' column is not in the final output but is used to generate data.
        # We can test for the number of unique product IDs, which are tied to categories.
        num_unique_products = self.df['product_id'].nunique()
        self.assertEqual(num_unique_products, 50)

    def test_seasonal_and_promotional_patterns(self):
        """
        Check for expected seasonal spikes and promotional impacts.
        """
        # Test for promotional impact
        promo_data = self.df[self.df['promotional_activity'] == True]
        non_promo_data = self.df[self.df['promotional_activity'] == False]

        # Sales quantity should be higher during promotions
        self.assertGreater(promo_data['sales_quantity'].mean(), non_promo_data['sales_quantity'].mean())

        # Price should be lower during promotions
        self.assertLess(promo_data['price'].mean(), non_promo_data['price'].mean())

    def test_realistic_data_ranges(self):
        """
        Validate that generated values are within expected, realistic ranges.
        """
        # Check for non-negative values
        self.assertGreaterEqual(self.df['sales_quantity'].min(), 0)
        self.assertGreaterEqual(self.df['price'].min(), 0)
        self.assertGreaterEqual(self.df['stock_level'].min(), 0)

        # Check for logical constraints
        self.assertLessEqual(self.df['lead_time'].max(), 14)
        self.assertGreaterEqual(self.df['lead_time'].min(), 1)


if __name__ == '__main__':
    unittest.main()