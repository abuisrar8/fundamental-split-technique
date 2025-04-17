from setuptools import setup, find_packages
from pathlib import Path

# The directory containing this file
HERE = Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="fst",                                 # Replace with your package name
    version="0.1.0",                            # Initial release
    description="Fundamental Split Technique: Performance-aware data splitting for robust regression",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fst",  # Update to your repo URL
    author="Your Name",                         # Update your name
    author_email="you@example.com",             # Update your email
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "pandas>=1.1",
        "scikit-learn>=0.24",
        "matplotlib>=3.3",
        "xgboost>=1.3",            # Optional; remove if you won’t use XGBRegressor
    ],
    include_package_data=True,
    entry_points={
        # If you want to provide a CLI, e.g. `fst-train`, you can uncomment and adapt:
        # 'console_scripts': [
        #     'fst-train = fst.core:main',
        # ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/fst/issues",
        "Source Code": "https://github.com/yourusername/fst",
    },
)
