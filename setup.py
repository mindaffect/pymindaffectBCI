from setuptools import setup, find_packages
import glob

with open("README.rst", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='mindaffectBCI',
      version='0.9.6',
      description='The MindAffect BCI python SDK',
      long_description_content_type='text/x-rst',      
      long_description=long_description,
      url='http://github.com/mindaffect/pymindaffectBCI',
      author='Jason Farquhar',
      author_email='jason@mindaffect.nl',
      license='MIT',
      packages=['mindaffectBCI','mindaffectBCI/examples/presentation','mindaffectBCI/examples/output'],#,find_packages(),#
      package_data={'mindaffectBCI':glob.glob('mindaffectBCI/*.txt')},
      include_package_data=True,
      #data_files=[('mindaffectBCI',glob.glob('mindaffectBCI/*.txt'))],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.5',      
      zip_safe=False)
