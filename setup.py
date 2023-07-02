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

def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop), open(project + '/__init__.py').read())
    return result.group(1)

def get_os_and_python_version():
    system = platform.system()
    python_version = ".".join(map(str, sys.version_info[:2]))
    if system.lower() == "windows":
        os_version = "windows"
    elif system.lower() == "darwin":
        os_version = "macos"
    elif system.lower() == "linux":
        os_version = platform.linux_distribution()[0].lower()
    else:
        os_version = "unknown"
    return os_version, python_version

os_version, python_version = get_os_and_python_version()

filename={  'windows':{ '3.7':'a_star.cp37-win_amd64.pyd',
                        '3.8':'a_star.cp38-win_amd64.pyd',
                        '3.9':'a_star.cp37-win_amd64.pyd',
                        '3.10':'a_star.cp310-win_amd64.pyd'},
            'macos'  :{ '3.7':'a_star.cpython-37m-darwin.so',
                        '3.8':'a_star.cpython-38-darwin.so',
                        '3.9':'a_star.cpython-39-darwin.so'},
            'ubuntu' :{ '3.7':'a_star.cpython-37m-x86_64-linux-gnu.so',
                        '3.8':'a_star.cpython-38-x86_64-linux-gnu.so',
                        '3.9':'a_star.cpython-39-x86_64-linux-gnu.so'}}

lib = filename[os_version][python_version]


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
    package_data={NAME: [lib]},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Mulan PSL v2',
    python_requires='>=3.7',
    classifiers=(
        'License :: OSI Approved :: Mulan Permissive Software License v2 (MulanPSL-2.0)',
        'Programming Language :: Python :: 3 :: Only',
        'Operating System :: OS Independent',
        'Environment :: GPU :: NVIDIA CUDA',
        'Environment :: Console',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Mathematics',
    ),
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
