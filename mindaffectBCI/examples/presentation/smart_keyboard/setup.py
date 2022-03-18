from setuptools import setup, find_packages

with open("requirements.txt", encoding='utf-8') as fh:
    install_requires = fh.read()
install_requires = install_requires.splitlines()

setup(
    name='MindAffect smart_keyboard',
    long_description=open('README.rst').read(),
    author='Thomas de Lange, Thomas Jurriaans, Damy Hillen, Joost Vossers, Jort Gutter, Florian Handke, Stijn Boosman',
    license='BSD-3',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=install_requires
)
