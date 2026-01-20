import os
import sys
from pathlib import Path

def setup_workspace(project_root: str = None) -> Path:
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False

    if project_root is not None:
        root = Path(project_root).resolve()
    else:
        root = Path.cwd().parent

    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    os.chdir(root)
    env = "Colab" if is_colab else "Local"
    print(f"Env: {env} | Root: {root}")
    return root
