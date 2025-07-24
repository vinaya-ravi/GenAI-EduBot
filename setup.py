from setuptools import setup, find_packages

setup(
    name="unt-gemma-agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "chainlit>=1.0.0",
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "openai==1.69.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.4.0",
        "numpy>=1.26.0"
    ],
) 