from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="nlp4web-codebase",
    version="0.0.0",
    author="Kexin Wang",
    author_email="kexin.wang.2049@gmail.com",
    description="Codebase of teaching materials for NLP4Web.",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://https://github.com/kwang2049/nlp4web-codebase",
    project_urls={
        "Bug Tracker": "https://github.com/kwang2049/nlp4web-codebase/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "nltk==3.8.1",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "pandas==2.2.2",
        "tqdm==4.66.5",
        "ujson==5.10.0",
        "joblib==1.4.2",
        "datasets==3.0.1",
        "pytrec_eval==0.5",
    ],
)
