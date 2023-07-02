dynamic_link = dict()
dynamic_link['windows'] = dict()
dynamic_link['macos'] = dict()
dynamic_link['ubuntu'] = dict()
dynamic_link['windows']['3.7'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1URS-PmytTx6vOjG8Kj5Ou-MSFna07w6x',
            '8d7b3622ece73d74ad82ddc2f821a598','a_star.cp37-win_amd64.pyd')
dynamic_link['windows']['3.8'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1_7Fjg-Ns9QRJBBfjVU9-e2wnvMZdfuA9',
            '7ef1a6473972790afea2639bf0713dd9','a_star.cp38-win_amd64.pyd')
dynamic_link['windows']['3.9'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1yEDF82lpd__V_UT_n8Su3m-IOHsIwH-U',
            'f1a90027fd5ba408631f06bb0ee76717','a_star.cp39-win_amd64.pyd')
dynamic_link['windows']['3.10'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=11okYNKQXlPY_5Pr2WUFsEvLyghGm8k4a',
            '57ad6057094d0f7b3946b68203206d72','a_star.cp310-win_amd64.pyd'),
dynamic_link['macos']['3.7'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1Xm2iOa_xyFEmpelt1Up_ocVOqG6IgFs7',
            '5e6b534440a49a4f5c8e39812dee33c9','a_star.cpython-37m-darwin.so')
dynamic_link['macos']['3.8'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1K3kBuWjO1oiL3ok0KdMQpK0pSwdUu-hEe',
            '55430a46cbdad0c64c1d978400b54c54','a_star.cpython-38-darwin.so')
dynamic_link['macos']['3.9'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=102q2i_o63jwjsuqljHt_9FgivkkB5UVd',
            '95a4aa75a5e679180d6e69f63d31a6ea','a_star.cpython-39-darwin.so')
dynamic_link['ubuntu']['3.7'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1UyfuIKy0_ow_4dxNrGsSFOdawkfgFWZL',
            '4d703eb2aeb885dcb0c945bcd48b3db0','a_star.cpython-37m-x86_64-linux-gnu.so')
dynamic_link['ubuntu']['3.8'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1QRsE55kH-gzHzjlLWZuH9J2UYx1kG0ca',
            '1a35e421fdf3270b6b3d4ca1bb17097b','a_star.cpython-38-x86_64-linux-gnu.so')
dynamic_link['ubuntu']['3.9'] = \
            ('https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1yXGWvBJQKSy_Ou-Eq5nKkjJd9UkFjzxA',
            '006330b49479f29c20b990ebd33baf25','a_star.cpython-39-x86_64-linux-gnu.so')


import platform
import sys
import pygmtools

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
python_version = '3.11'
print("Current OS: ", os_version)
print("Current Python version: ", python_version)

try:
    url, md5, filename = dynamic_link[os_version][python_version][0]
except:
    if os_version == 'windows':
        url, md5, filename  = dynamic_link[os_version]['3.10'][0]
    else:
        url, md5, filename  = dynamic_link[os_version]['3.9'][0]


print(filename)
print(url)
print(md5)
pygmtools.utils.download(filename, url, md5, to_cache=False)

