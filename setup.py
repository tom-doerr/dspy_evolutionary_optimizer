from setuptools import find_packages, setup


setup(
    name="evoprompt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "dspy-ai",
        "matplotlib",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-timeout",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for evolving prompts with numeric feedback",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/evoprompt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
