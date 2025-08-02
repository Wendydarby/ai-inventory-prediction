"""Unit tests for data preprocessing module."""

import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing import load_data, clean_data, create_features, normalize_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_data = pd.DataFrame({
            'product_id': ['SKU001', 'SKU001', 'SKU002', 'SKU002'],
            'date': pd.date_range(start='2024-01-01', periods=4),
            'sales_quantity': [100, np.nan, 150, 200],
            'price': [10.0, 10.0, 15.0, 15.0],
            'promotional_activity': [True, False, True, False],
            'weather_data': [25.0, 24.0, np.nan, 26.0],
            'stock_level': [500, 400, 600, 400],
            'lead_time': [5, 5, 7, 7]
        })
        
    def test_load_data(self):
        """Test data loading functionality"""
        # Save test data to temp file
        self.test_data.to_csv('test_inventory.csv', index=False)
        loaded_data = load_data('test_inventory.csv')
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(len(loaded_data), 4)
        
    def test_clean_data(self):
        """Test data cleaning functionality"""
        cleaned_data = clean_data(self.test_data)
        # Check no missing values
        self.assertFalse(cleaned_data.isnull().any().any())
        # Check original row count maintained
        self.assertEqual(len(cleaned_data), len(self.test_data))
        
    def test_create_features(self):
        """Test feature engineering"""
        featured_data = create_features(self.test_data)
        expected_features = ['day_of_week', 'month', 'is_weekend', 'sales_lag_1']
        for feature in expected_features:
            self.assertIn(feature, featured_data.columns)
            
    def test_normalize_data(self):
        """Test data normalization"""
        normalized_data = normalize_data(self.test_data)
        # Check numerical columns are scaled between 0 and 1
        numeric_cols = ['sales_quantity', 'price', 'weather_data', 'stock_level']
        for col in numeric_cols:
            self.assertTrue(normalized_data[col].between(0, 1).all())

if __name__ == '__main__':
    unittest.main()