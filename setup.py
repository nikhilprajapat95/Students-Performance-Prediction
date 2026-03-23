from ast import List

from setuptools import setup, find_packages, setup

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='Student Performance', 
    version='1.0.0',   
    author='Nikhil',
    author_email='nikhilprajapat955@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)