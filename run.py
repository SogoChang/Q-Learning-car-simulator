import tkinter as tk
import os
import sys
import importlib.util
from tkinter import messagebox

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

# Change working directory to the application path
os.chdir(application_path)

def check_dependencies():
    """Check if all required modules are installed"""
    required_modules = [
        'tkinter', 'matplotlib', 'numpy', 'pickle', 'threading'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        return False, missing_modules
    return True, []

def check_required_files():
    """Check if the required files exist"""
    required_files = [
        'claude_playground.py',
        'qlearning.py',
        'UI.py',
        'simple_geometry.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        return False, missing_files
    return True, []

if __name__ == "__main__":
    '''
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"Error: Missing required modules: {', '.join(missing_deps)}")
        print("Please install the missing modules and try again.")
        sys.exit(1)
    
    # Check required files
    files_ok, missing_files = check_required_files()
    if not files_ok:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        print("Please make sure all required files are in the same directory.")
        sys.exit(1)
    '''
    # Import the UI module
    try:
        # First import the playground and qlearning modules to ensure they load correctly
        import playground
        import qlearning
        
        # Then import our UI module
        from UI import QlearningUI
        
        # Create and start the UI
        root = tk.Tk()
        app = QlearningUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting the UI: {str(e)}")
        
        # If we have a GUI, show error in messagebox
        try:
            messagebox.showerror("Error", f"Failed to start the application:\n{str(e)}")
        except:
            pass  # If messagebox fails, we already printed to console
        
        sys.exit(1)