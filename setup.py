from setuptools import setup, find_packages
import glob

with open("README.rst", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='mindaffectBCI',
      version='0.9.24',
      description='The MindAffect BCI python SDK',
      long_description_content_type='text/x-rst',      
      long_description=long_description,
      url='http://github.com/mindaffect/pymindaffectBCI',
      author='Jason Farquhar',
      author_email='jason@mindaffect.nl',
      license='MIT',
      packages=['mindaffectBCI','mindaffectBCI/decoder','mindaffectBCI/decoder/offline','mindaffectBCI/examples/presentation','mindaffectBCI/examples/output','mindaffectBCI/examples/utilities','mindaffectBCI/examples/acquisition'],#,find_packages(),#
      include_package_data=True,
      package_data={'mindaffectBCI':glob.glob('mindaffectBCI/*.txt'), 
                    'mindaffectBCI':glob.glob('mindaffectBCI/*.png'), 
                    'mindaffectBCI':glob.glob('mindaffectBCI/*.json'), 
                    'mindaffectBCI.hub':glob.glob('mindaffectBCI/hub/*'), 
                    'mindaffectBCI.decoder':glob.glob('mindaffectBCI/decoder/*.pk'), 
                    'mindaffectBCI.examples.presentation':glob.glob('mindaffectBCI/examples/presentation/*.txt')},
      data_files=[('mindaffectBCI',glob.glob('mindaffectBCI/*.png')), 
                  ('mindaffectBCI',glob.glob('mindaffectBCI/*.txt')), 
                  ('mindaffectBCI',glob.glob('mindaffectBCI/*.json')), 
                  ('mindaffectBCI/hub',glob.glob('mindaffectBCI/hub/*')), 
                  ('mindaffectBCI/decoder',glob.glob('mindaffectBCI/decoder/*.pk')), 
                  ('mindaffectBCI.examples.presentation',glob.glob('mindaffectBCI/examples/presentation/*.txt'))],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.5',
      install_requires=['numpy>=1.0.2', 'pyglet>=1.2', 'scipy>=1.0', 'brainflow>=3.0',
'matplotlib>=3.0'],
      #entry_points={ 'console_scripts':['online_bci=mindaffectBCI.online_bci']},
      zip_safe=False)
