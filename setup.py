# /var/www/html/GIT PROJECTS/PYTHON GITHUB CONTRIBUTION PROJECTS/dataprep_lite/setup.py

from setuptools import setup, find_packages
import re
import os

# Function to read the version from dataprep_lite/dataprep_lite/__init__.py
def get_version():
    # Path to the __init__.py file of your *package*
    # This is now inside the 'dataprep_lite' sub-directory.
    package_init_py = os.path.join(os.path.dirname(__file__), "dataprep_lite", "__init__.py")
    with open(package_init_py, "r") as f:
        version_file_content = f.read()

    version_match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]*)['\"]",
        version_file_content,
        re.M,
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string in %s." % (package_init_py,))

# Read README.md for the long description
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "A lightweight data cleaning and preprocessing library for Python."

setup(
    name='dataprep-lite', # The name of your package on PyPI (can have hyphens)
    version=get_version(), # Call corrected get_version()
    author='Rahul K',
    author_email='rkolekar913@gmail.com', # Removed the unnecessary brackets
    description='A lightweight data cleaning and preprocessing library for Python.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rahulkolekardev/DataPrep-Lite',
    license='MIT',

    # find_packages will look for packages in the current directory (where setup.py is)
    # It will find the 'dataprep_lite' directory (because it has an __init__.py)
    # and then recursively find sub-packages within it (cleaning, core, preprocessing).
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'examples.*']),

    # If your actual package 'dataprep_lite' was inside an 'src' directory
    # (e.g., src/dataprep_lite), you would use:
    # package_dir={'': 'src'},
    # packages=find_packages(where='src', exclude=['tests', 'tests.*', 'examples', 'examples.*']),
    # But given your current structure, the above find_packages is correct.

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
            'build', # Add 'build' to dev dependencies
        ],
    },
)