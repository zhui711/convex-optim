"""从源码直接运行，无需先安装本项目。依赖为 NumPy 和 Matplotlib。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from convex_optim.run_all import main

if __name__ == "__main__":
    main()
