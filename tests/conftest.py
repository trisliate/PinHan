import sys
from pathlib import Path

# 将项目根加入 sys.path，方便 tests 用包路径导入模块
sys.path.insert(0, str(Path(__file__).parent.parent))
