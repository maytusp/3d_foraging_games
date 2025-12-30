import pybullet as p
import pkgutil
import sys

def test_egl():
    print("1. Connecting to PyBullet (DIRECT mode)...")
    p.connect(p.DIRECT)

    print("2. Searching for EGL plugin...")
    egl = pkgutil.get_loader('eglRenderer')
    
    if egl:
        plugin_path = egl.get_filename()
        print(f"   Found plugin at: {plugin_path}")
        
        print("3. Attempting to load plugin...")
        plugin_id = p.loadPlugin(plugin_path, "_eglRendererPlugin")
        
        if plugin_id < 0:
            print("\n❌ FAILURE: Plugin found but failed to load.")
            print("   Possible cause: No GPU allocated or driver mismatch.")
        else:
            print(f"\n✅ SUCCESS: EGL Plugin loaded! (ID: {plugin_id})")
            print("   Hardware acceleration is active.")
    else:
        print("\n❌ FAILURE: Python could not find 'eglRenderer'.")
        print("   Try: pip install pybullet --upgrade")

if __name__ == "__main__":
    test_egl()