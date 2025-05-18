# /var/www/html/GIT PROJECTS/PYTHON GITHUB CONTRIBUTION PROJECTS/dataprep_lite/setup.py

from setuptools import setup, find_packages
import re # Import the regular expression module
import os

# MORE ROBUST Function to read the version from __init__.py
def get_version():
    # __init__.py is in the same directory as setup.py
    init_py_path = "__init__.py"
    with open(init_py_path, "r") as f:
        version_file_content = f.read()
    
    # Use a regular expression to find the __version__ string
    # This regex looks for a line like: __version__ = "0.1.0"
    # or __version__ = '0.1.0'
    version_match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]*)['\"]",
        version_file_content,
        re.M, # M for multiline matching
    )
    if version_match:
        return version_match.group(1) # Return the captured version string
    raise RuntimeError("Unable to find version string in %s." % (init_py_path,))


# Read README.md for the long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A lightweight data cleaning and preprocessing library for Python."

setup(
    name='dataprep-lite',
    version=get_version(), # Call corrected get_version()
    author='Rahul K',                # Your name
    author_email='[rkolekar913@gmail.com]', # Your email
    description='A lightweight data cleaning and preprocessing library for Python.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rahulkolekardev/DataPrep-Lite',
    license='MIT',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
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
    keywords='data cleaning, data preprocessing, pandas, feature engineering, data science, machine learning, etl',
    extras_require={
        'dev': [
            'pytest>=6.0',
            'twine>=3.0.0',
            'wheel',
        ],
    },
)