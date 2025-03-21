import os
import numpy as np

# Path to the problematic file
squeeze_pro_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), '.pythonlibs/lib/python3.11/site-packages/pandas_ta/momentum/squeeze_pro.py')

# Check if the file exists
if os.path.exists(squeeze_pro_path):
    # Read the file
    with open(squeeze_pro_path, 'r') as file:
        content = file.read()

    # Replace the problematic import
    content = content.replace('from numpy import NaN as npNaN', 'import numpy as np\nnpNaN = np.nan')

    # Write the changes back
    with open(squeeze_pro_path, 'w') as file:
        file.write(content)
    
    print("Fixed pandas_ta library for compatibility with current numpy version.")
else:
    print(f"Could not find the file at: {squeeze_pro_path}")
    import site
    print("Installed packages location:", site.getsitepackages())