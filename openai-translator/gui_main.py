import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # 当前目录就是项目根目录
sys.path.append(project_root)

from gui import main

if __name__ == "__main__":
    main() 