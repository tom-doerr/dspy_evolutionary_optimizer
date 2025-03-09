"""
Debug script to investigate DSPy module structure
"""

import sys
import importlib
import inspect
import os

def main():
    print("=== DSPy Debug Information ===")
    
    # Try to import dspy
    try:
        import dspy
        print(f"DSPy imported successfully")
        print(f"DSPy version: {getattr(dspy, '__version__', 'unknown')}")
        print(f"DSPy module location: {inspect.getfile(dspy)}")
        
        # Check for LM attribute
        if hasattr(dspy, 'LM'):
            print("dspy.LM exists")
            print(f"dspy.LM type: {type(dspy.LM)}")
        else:
            print("dspy.LM does NOT exist")
        
        # List all top-level attributes
        print("\nTop-level DSPy attributes:")
        for attr in sorted(dir(dspy)):
            if not attr.startswith('_'):
                print(f"  {attr}")
        
        # Check for models module
        if hasattr(dspy, 'models'):
            print("\ndspy.models exists")
            print("Contents of dspy.models:")
            for attr in sorted(dir(dspy.models)):
                if not attr.startswith('_'):
                    print(f"  {attr}")
        
        # Check Python path
        print("\nPython path:")
        for path in sys.path:
            print(f"  {path}")
        
        # Check for multiple dspy installations
        print("\nChecking for multiple DSPy installations:")
        dspy_locations = []
        for path in sys.path:
            potential_dspy = os.path.join(path, 'dspy')
            if os.path.exists(potential_dspy):
                dspy_locations.append(potential_dspy)
        
        if len(dspy_locations) > 1:
            print(f"WARNING: Multiple DSPy installations found:")
            for loc in dspy_locations:
                print(f"  {loc}")
        elif dspy_locations:
            print(f"Single DSPy installation found at: {dspy_locations[0]}")
        else:
            print("No DSPy installation directory found in sys.path")
            
    except ImportError as e:
        print(f"Failed to import dspy: {e}")
    
    # Try to find LM in any submodule
    try:
        import dspy
        print("\nSearching for LM in DSPy submodules:")
        for module_name in dir(dspy):
            if not module_name.startswith('_'):
                try:
                    module = getattr(dspy, module_name)
                    if hasattr(module, 'LM'):
                        print(f"  Found LM in dspy.{module_name}")
                except Exception as e:
                    pass
    except:
        pass

if __name__ == "__main__":
    main()
