import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GuruDiff", 
    version="1.0.1",
    author="Erick Ruiz, Jingyuan Liu, Simon (Xin) Dong, Kailas Amin",
    author_email="eruiz@g.harvard.edu, jingyuanliu@g.harvard.edu, xindong@g.harvard.edu, kailasamin@college.harvard.edu",
    description="A easy-used auto-differentiation package, supporting both forward mode and reverse mode",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://https://github.com/Software-Samurais/cs207-FinalProject",
    install_requires = [
        'numpy==1.17.3',
        'pytest==5.2.2'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)