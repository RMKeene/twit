import setuptools
print("Setup...")
with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="twit",
	version="0.1.1",
	author="Richard Keene",
	author_email="rmkeene@gmail.com",
	description="Tensor Weighted Interpolative Transfer",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/RMKeene/twit",
	packages=setuptools.find_packages(),
	data_files=[('lib\\',["x64\\Release\\twitc.lib"]), ('libd\\',["x64\\Debug\\twitc.lib"])],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.6',
)