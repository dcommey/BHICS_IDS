from setuptools import setup, find_packages

setup(
    name="bhics_ids",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'scikit-learn>=0.24.0',
        'tensorflow>=2.8.0',
        'xgboost>=1.5.0',
        'seaborn>=0.11.0',
        'psutil>=5.8.0',
        'joblib>=1.0.0',
    ],
)