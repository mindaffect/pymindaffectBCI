from setuptools import setup, find_packages
import glob

with open("README.rst", encoding='utf-8') as fh:
    long_description = fh.read()
with open("requirements.txt", encoding='utf-8') as fh:
    install_requires = fh.read()
install_requires = install_requires.splitlines()

setup(name='mindaffectBCI',
      version='0.10.4',
      description='The MindAffect BCI python SDK',
      long_description_content_type='text/x-rst',      
      long_description=long_description,
      url='http://github.com/mindaffect/pymindaffectBCI',
      author='Jason Farquhar',
      author_email='jason@mindaffect.nl',
      license='MIT',
      packages=['mindaffectBCI',
                'mindaffectBCI/decoder',
                'mindaffectBCI/decoder/offline',
                'mindaffectBCI/examples/presentation',
                'mindaffectBCI/examples/presentation/smart_keyboard',
                'mindaffectBCI/examples/presentation/smart_keyboard/windows',
                'mindaffectBCI/examples/output',
                'mindaffectBCI/examples/utilities',
                'mindaffectBCI/examples/acquisition'],#,find_packages(),#
      include_package_data=True,
      package_data={'mindaffectBCI.config':glob.glob('mindaffectBCI/config/*.json'),
                    'mindaffectBCI.stimulus_sequence':glob.glob('mindaffectBCI/stimulus_sequence/*.txt'), 
                    'mindaffectBCI.stimulus_sequence':glob.glob('mindaffectBCI/stimulus_sequence/*.png'), 
                    'mindaffectBCI.hub':glob.glob('mindaffectBCI/hub/*'), 
                    'mindaffectBCI.examples.presentation.symbols':glob.glob('mindaffectBCI/examples/presentation/symbols/*.txt'),
                    'mindaffectBCI.examples.presentation.smart_keyboard.keypad_layouts':glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/keypad_layouts/*.json'),
                    'mindaffectBCI.examples.presentation.smart_keyboard.configs':glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/configs/*.json'),
                    'mindaffectBCI.examples.presentation.smart_keyboard.key_icons':glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/key_icons/*.png'),
                    'mindaffectBCI.examples.presentation.smart_keyboard.dictionaries.frequency_lists':glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/dictionaries/frequency_lists/*.txt'),
                    },
      data_files=[('',('requirements.txt',)),
                  ('mindaffectBCI/config',glob.glob('mindaffectBCI/config/*.json')),
                  ('mindaffectBCI/stimulus_sequence',glob.glob('mindaffectBCI/stimulus_sequence/*.txt')), 
                  ('mindaffectBCI/stimulus_sequence',glob.glob('mindaffectBCI/stimulus_sequence/*.png')), 
                  ('mindaffectBCI/hub',glob.glob('mindaffectBCI/hub/*.jar')), 
                  ('mindaffectBCI/decoder',glob.glob('mindaffectBCI/decoder/*.pk')), 
                  ('mindaffectBCI/examples/presentation/symbols',glob.glob('mindaffectBCI/examples/presentation/symbols/*.txt')),
                  ('mindaffectBCI.examples.presentation.smart_keyboard.keypad_layouts',glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/keypad_layouts/*.json')),
                  ('mindaffectBCI.examples.presentation.smart_keyboard.configs',glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/configs/*.json')),
                  ('mindaffectBCI.examples.presentation.smart_keyboard.key_icons',glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/key_icons/*.png')),
                  ('mindaffectBCI.examples.presentation.smart_keyboard.dictionaries.frequency_lists',glob.glob('mindaffectBCI/examples/presentation/smart_keyboard/dictionaries/frequency_lists/*.txt')),
                  ],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.5',
      install_requires=install_requires,
      #entry_points={ 'console_scripts':['online_bci=mindaffectBCI.online_bci']},
      zip_safe=False)
