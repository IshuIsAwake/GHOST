from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as req:
        return [line.strip() for line in req if line.strip() and not line.startswith('#')]

setup(
    name="ghost-net",
    version="1.0.0",
    description="GHOST: Generalizable Hyperspectral Observation Segmentation Tool",
    author="Hackathon Team",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'ghost=ghost.cli:main',
        ],
    },
    python_requires='>=3.8',
)
