from setuptools import setup, find_packages

setup(
    name="ai-inventory-prediction",
    version="0.1.0",
    description="AI-Powered Inventory Prediction System",
    author="Wendy Darby, Thomas Hudson"
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "keras",
        "plotly",
        "statsmodels"
    ],
)