from setuptools import setup, find_packages

def require(require):
    with open(require) as file:
        requirement=file.readlines()
    requirement=[package.strip() for package in requirement]
    if "-e ." in requirement:
        requirement.remove("-e .")
    return requirement

setup(
    name="Text Emoji Fusion",
      version="1.0.0",
      author="Adesegun Oluwademilade Daniel",
      author_email="adesegundaniel11@gmail.com",
      packages=find_packages(),
      install_requires=require('requirements.txt'))


