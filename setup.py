import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml_algos",
    version="0.1",
    author="",
    description="PyTorch implementations of the most popular ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=["ml_algos"],
    python_requires=">=3.11",
)