import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlmc-eig-grad",
    version="1.0.0",
    author="Takashi Goda, Tomohiko Hironaka, Wataru Kitade",
    author_email="goda@frcer.t.u-tokyo.ac.jp, "
    + "hironaka-tomohiko@g.ecc.u-tokyo.ac.jp, "
    + "kitade-wataru114@g.ecc.u-tokyo.ac.jp",
    description="optimization of EIG using MLMC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Goda-Research-Group/MLMC_stochastic_gradient",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": ["mlmc_experiments=mlmc_eig_grad.experiments:main",]
    },
    python_requires=">=3.6.0",
)
