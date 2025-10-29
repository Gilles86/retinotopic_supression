
from setuptools import setup, find_packages

setup(
    name="retsupp",
    version="0.1",
    packages=find_packages(),
    description="A 7T project by the Theeuwes group at the Vrije Universiteit Amsterdam",
    author="Gilles de Hollander",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/retsupp",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Add your project dependencies here
    ],
    package_data={
        "retsupp": ["data/subjects.yml"],
    },
    include_package_data=True,
)
