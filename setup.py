import setuptools
print("Setup...")
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="twit-rkeene",
    version="0.0.1",
    author="Richard Keene",
    author_email="rmkeene@gmail.com",
    description="Tensor Weighted Interpolative Transfer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/twit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)