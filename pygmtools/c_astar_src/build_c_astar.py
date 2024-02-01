import os
import glob
import shutil

ori_dir = os.getcwd()
os.chdir('pygmtools/c_astar_src')

try:
    os.system("python c_astar_setup.py build_ext --inplace")
except:
    os.system("python3 c_astar_setup.py build_ext --inplace")
      
current_dir = os.getcwd()

ext_files = glob.glob(os.path.join(current_dir, '*.pyd')) + \
            glob.glob(os.path.join(current_dir, '*.so'))

if len(ext_files) == 0:
    raise ValueError("there is no .pyd or .so")
elif len(ext_files) > 1:
    raise ValueError("too many files end with .pyd or .so")
else:
    target_dir = os.path.abspath(os.path.join('..'))
    shutil.copy(ext_files[0], target_dir)
    
os.chdir(ori_dir)