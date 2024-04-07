import os
import shutil


def safe_create_directory(path):
    try:
        if os.path.exists(path):
            print(f"目录 {path} 已存在，将被删除。")
            shutil.rmtree(path)

        os.mkdir(path)
        print(f"目录 {path} 创建成功。")
    except Exception as e:
        print(f"创建目录失败: {e}")
        raise


if __name__ == '__main__':
    safe_create_directory('./11')
