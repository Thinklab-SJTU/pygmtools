import tarfile
import os

def tar_folder(folder_path, tar_file_name):
    """
    将指定文件夹中的文件压缩成 tar 文件

    :param folder_path: 要压缩的文件夹路径
    :param tar_file_name: 压缩文件的名称
    """
    # 创建 TarFile 对象
    with tarfile.open(tar_file_name, 'w:gz') as tarObj:
        # 获取文件夹中的所有文件
        for filename in os.listdir(folder_path):
            # 构造文件的绝对路径
            file_path = os.path.join(folder_path, filename)
            # 如果是文件，则将其添加到压缩文件中
            if os.path.isfile(file_path):
                tarObj.add(file_path, arcname=filename)

tar_folder('a_star','a_star.tar.gz')