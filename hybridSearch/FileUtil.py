from pathlib import Path
import shutil
import os

def delete_file_if_exists(file_path):
    """
    检查文件路径是否存在，若存在则尝试删除。

    :param file_path: 文件路径（字符串）
    :return: True（成功删除或文件不存在） or False（文件存在但删除失败）
    """
    path = Path(file_path)
    if(file_path):
        if path.exists():
            try:
                path.unlink()  # 删除文件
                return True
            except OSError:
                return False
    return True

def delete_directory_and_contents(directory_path):
    """
    递归删除指定文件夹及其所有子文件和子文件夹。

    :param directory_path: 要删除的文件夹路径（字符串）
    :return: True（成功删除或文件夹不存在） or False（文件夹存在但删除失败）
    """
    if os.path.exists(directory_path):
        try:
            shutil.rmtree(directory_path)  # 递归删除目录及内容
            return True
        except Exception as e:
            print(f"删除目录失败: {e}")
            return False
    return True  # 文件夹不存在视为成功