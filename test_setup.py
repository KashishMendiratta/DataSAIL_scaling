"""Quick test to verify all dependencies are working."""
import sys

print("Testing imports...\n")

success_count = 0
total_count = 0

def test_import(module_name, import_statement, display_name=None):
    global success_count, total_count
    total_count += 1
    display = display_name or module_name
    try:
        exec(import_statement)
        print(f"✓ {display}")
        success_count += 1
        return True
    except ImportError as e:
        print(f"✗ {display}: {e}")
        return False

# Test core packages
test_import("numpy", "import numpy as np")
test_import("pandas", "import pandas as pd")
test_import("sklearn", "import sklearn", "scikit-learn")
test_import("matplotlib", "import matplotlib.pyplot as plt")
test_import("seaborn", "import seaborn as sns")

# Test molecular/ML packages
test_import("rdkit", "from rdkit import Chem", "RDKit")
test_import("datasail", "import datasail", "DataSAIL")
test_import("xgboost", "import xgboost as xgb", "XGBoost")
test_import("lightgbm", "import lightgbm as lgb", "LightGBM")

print(f"\n{'='*50}")
print(f"Results: {success_count}/{total_count} packages imported successfully")
print(f"{'='*50}\n")

if success_count == total_count:
    print("✓ All packages working! You're ready to start.")
    
    # Show versions
    print("\nPackage versions:")
    import numpy as np
    import pandas as pd
    import sklearn
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  NumPy: {np.__version__}")
    print(f"  Pandas: {pd.__version__}")
    print(f"  scikit-learn: {sklearn.__version__}")
    try:
        import datasail
        print(f"  DataSAIL: {datasail.__version__}")
    except:
        pass
else:
    print("⚠ Some packages failed to import. See errors above.")
    print("Try running: pip install <package-name>")
