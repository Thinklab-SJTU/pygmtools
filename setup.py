#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree
import re
import platform
from setuptools import find_packages, setup, Command
import tarfile
import distro
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
import shutil

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

# Package meta-data.
NAME = 'pygmtools'
DESCRIPTION = 'pygmtools provides graph matching solvers in Python API and supports numpy and pytorch backends. ' \
              'pygmtools also provides dataset API for standard graph matching benchmarks.'
URL = 'https://pygmtools.readthedocs.io/'
AUTHOR = get_property('__author__', NAME)
VERSION = get_property('__version__', NAME)

REQUIRED = [
     'requests>=2.25.1', 'scipy>=1.4.1', 'Pillow>=7.2.0', 'numpy>=1.18.5', 'easydict>=1.7', 'appdirs>=1.4.4', 'tqdm>=4.64.1','wget>=3.2'
]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION



def get_os_and_python_version():
    system = platform.system()
    python_version = ".".join(map(str, sys.version_info[:2]))
    if system.lower() == "windows":
        os_version = "windows"
    elif system.lower() == "darwin":
        os_version = "macos"
    elif system.lower() == "linux":
        os_version = distro.name().lower()
    else:
        raise ValueError("Unknown System")
    if (python_version == '3.11'):
        python_version = '3.10'
    return os_version, python_version
        
def untar_file(tar_file_path, extract_folder_path):
    with tarfile.open(tar_file_path, 'r:gz') as tarObj:
        tarObj.extractall(extract_folder_path)

if os.path.exists(os.path.join(NAME,'lib/a_star.tar.gz')):
    untar_file(os.path.join(NAME,'lib/a_star.tar.gz'),os.path.join(NAME,'lib'))
    
filename={'windows':{ '3.7':'a_star.cp37-win_amd64.pyd',
                        '3.8':'a_star.cp38-win_amd64.pyd',
                        '3.9':'a_star.cp39-win_amd64.pyd',
                        '3.10':'a_star.cp310-win_amd64.pyd'},
            'macos'  :{ '3.7':'a_star.cpython-37m-darwin.so',
                        '3.8':'a_star.cpython-38-darwin.so',
                        '3.9':'a_star.cpython-39-darwin.so',
                        '3.10':'a_star.cpython-310-darwin.so'},
            'ubuntu' :{ '3.7':'a_star.cpython-37m-x86_64-linux-gnu.so',
                        '3.8':'a_star.cpython-38-x86_64-linux-gnu.so',
                        '3.9':'a_star.cpython-39-x86_64-linux-gnu.so',
                        '3.10':'a_star.cpython-310-x86_64-linux-gnu.so'}}

class CustomBdistWheelCommand(_bdist_wheel):
    def run(self):
        # os_version, python_version = get_os_and_python_version()
        # dynamic_link = filename[os_version][python_version]
        # shutil.copy2(os.path.join(NAME, 'lib', dynamic_link), os.path.join(NAME, dynamic_link))
        # shutil.rmtree(os.path.join(NAME, 'lib'))
        # if os.path.exists(os.path.join(NAME,"a_star.tar.gz")):
        #     os.remove(os.path.join(NAME,"a_star.tar.gz"))
        _bdist_wheel.run(self)
        

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    package_data={NAME: ['lib/*','*.pyd','*.so']},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Mulan PSL v2',
    python_requires='>=3.7',
    classifiers=[
        'License :: OSI Approved :: Mulan Permissive Software License v2 (MulanPSL-2.0)',
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: Console',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        'bdist_wheel': CustomBdistWheelCommand,
    },
)
