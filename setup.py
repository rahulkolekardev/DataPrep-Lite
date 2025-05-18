
from setuptools import setup, find_packages
import os

# Function to read the version from dataprep_lite/__init__.py
def get_version(package_name):
    version = {}
    # Correctly construct the path to __init__.py relative to setup.py
    init_py_path = os.path.join(package_name, "__init__.py")
    with open(init_py_path) as fp:
        exec(fp.read(), version)
    return version['__version__']

# Read README.md for the long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A lightweight data cleaning and preprocessing library for Python."

setup(
    name='dataprep-lite',  # The name of your package on PyPI
    version=get_version('dataprep_lite'), # Dynamically get version from your package
    author='Rahul K',                # Your name
    author_email='[rkolekar913@gmail.com]', # Your email
    description='A lightweight data cleaning and preprocessing library for Python.',
    long_description=long_description,
    long_description_content_type="text/markdown", # Important for PyPI to render README.md
    url='https://github.com/rahulkolekardev/DataPrep-Lite', # Your project's GitHub URL
    license='MIT',  # Your chosen license (make sure a LICENSE file exists)
    
    # find_packages() will automatically discover your 'dataprep_lite' package
    # and its sub-packages (core, cleaning, preprocessing).
    # Exclude 'tests' and 'examples' from the installed package.
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    
    # List of runtime dependencies
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
    ],
    
    # Specify compatible Python versions
    python_requires='>=3.8',
    
    # Classifiers help users find your project by browsing PyPI
    # Full list: https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha', # Or '4 - Beta', '5 - Production/Stable'
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
    
    # Keywords that describe your package
    keywords='data cleaning, data preprocessing, pandas, feature engineering, data science, machine learning, etl',
    
    # (Optional) For development dependencies like pytest:
    # These can be installed using: pip install -e ".[dev]"
    extras_require={
        'dev': [
            'pytest>=6.0',
            'twine>=3.0.0', # For uploading to PyPI
            'wheel',        # For building wheel distributions
            # 'sphinx',     # If you add Sphinx for documentation
            # 'black',      # For code formatting
            # 'flake8',     # For linting
        ],
    },
    
    # (Optional) If your package has scripts to be installed in the system path
    # entry_points={
    #     'console_scripts': [
    #         'my-package-command=my_package.cli:main',
    #     ],
    # },
)
