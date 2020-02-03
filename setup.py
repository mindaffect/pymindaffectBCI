from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='mindaffectBCI',
      version='0.9',
      description='The MindAffect BCI python SDK',
      long_description=long_description,
      long_description_content_type="text/markdown",      
      url='http://github.com/mindaffect/pymindaffectBCI',
      author='Jason Farquhar',
      author_email='jason@mindaffect.nl',
      license='MIT',
      packages=['mindaffectBCI'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.5',      
      zip_safe=False)