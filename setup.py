from setuptools import setup

with open("README.rst", encoding='utf-8') as fh:
    long_description = fh.read()
print(long_description)

setup(name='mindaffectBCI',
      version='0.9.1',
      description='The MindAffect BCI python SDK',
      long_description_content_type='text/x-rst',      
      long_description=long_description,
      url='http://github.com/mindaffect/pymindaffectBCI',
      author='Jason Farquhar',
      author_email='jason@mindaffect.nl',
      license='MIT',
      packages=['mindaffectBCI'],
      #package_data={'mindaffectBCI':['codebooks/*.txt']},
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.5',      
      zip_safe=False)
