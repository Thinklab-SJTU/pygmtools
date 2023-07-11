import platform
import tarfile
import distro
import sys
import os

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

NAME = 'pygmtools'

try:
    from pygmtools.a_star import a_star
except ModuleNotFoundError:
    os_version, python_version = get_os_and_python_version()
    dynamic_link = filename[os_version][python_version]
    if not os.path.exists(os.path.join(NAME,dynamic_link)):
        untar_file(os.path.join(NAME,'a_star.tar.gz'),NAME)
        for os_version in filename.keys():
            for python_version in filename[os_version].keys():
                if filename[os_version][python_version] != dynamic_link:
                    os.remove(os.path.join(NAME,filename[os_version][python_version]))