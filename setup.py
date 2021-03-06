from setuptools import setup, find_packages

VERSION = "0.1.0"

def readme():
    try:
        with open("README.md") as f:
            return f.read()
    except IOError:
        return ""


setup(
    name="Buteo_eo",
    version="0.1.1",
    author="Casper Fibaek",
    author_email="casperfibaek@gmail.com",
    description="A pythonic way of working with EO data for Deep Learning",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/casperfibaek/buteo_eo",
    project_urls={
        "Bug Tracker": "https://github.com/casperfibaek/buteo/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Alpha",
    ],
    packages=find_packages(),
    zip_safe=True,
    install_requires=[],
    include_package_data=True,
)
