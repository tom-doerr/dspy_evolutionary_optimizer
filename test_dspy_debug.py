"""
Debug script to investigate DSPy module structure
"""

import sys
import importlib
import os
import pkgutil

def main():
    print("=== DSPy Debug Information ===")
    
    # Try to import dspy
    try:
        import dspy
        print("DSPy imported successfully")
        print(f"DSPy version: {getattr(dspy, '__version__', 'unknown')}")
        
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
        
        # Check installed packages
        print("\nInstalled packages:")
        try:
            import pkg_resources
            for dist in pkg_resources.working_set:
                if 'dspy' in dist.project_name.lower():
                    print(f"  {dist.project_name} {dist.version}")
        except ImportError:
            print("  pkg_resources not available")
        
        # Check for simpledspy
        try:
            import simpledspy
            print("\nsimpledspy is installed")
            print(f"simpledspy version: {getattr(simpledspy, '__version__', 'unknown')}")
            print("simpledspy attributes:")
            for attr in sorted(dir(simpledspy)):
                if not attr.startswith('_'):
                    print(f"  {attr}")
                    
            # Check if simpledspy has LM
            if hasattr(simpledspy, 'LM'):
                print("simpledspy.LM exists")
        except ImportError:
            print("\nsimpledspy is NOT installed")
        
        # Check if dspy is a namespace package
        spec = importlib.util.find_spec('dspy')
        if spec:
            print(f"\nDSPy spec: {spec}")
            print(f"DSPy spec origin: {spec.origin}")
            print(f"DSPy spec submodule_search_locations: {spec.submodule_search_locations}")
            
            if spec.submodule_search_locations:
                print("\nDSPy submodules:")
                for location in spec.submodule_search_locations:
                    if os.path.exists(location):
                        for _, name, ispkg in pkgutil.iter_modules([location]):
                            print(f"  {name} ({'package' if ispkg else 'module'})")
        
        # Check Python path
        print("\nPython path (first 5 entries):")
        for path in sys.path[:5]:
            print(f"  {path}")
            
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
                except Exception:
                    pass
    except:
        pass

if __name__ == "__main__":
    main()
