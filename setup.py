from setuptools import setup, find_packages

setup(
    name="house_prices",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
    ],
)
