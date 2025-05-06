from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="ANNEVO",
    version="1.0",
    author="Pengyu Zhang",
    author_email="pengyuzhang@stu.xjtu.edu.cn",
    description="Ab initio gene prediction tool",
    long_description=open("README.md", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PyZhang-Bio/ANNEVO",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Bioinformatics',
    ],
    python_requires='>=3.6',
)
