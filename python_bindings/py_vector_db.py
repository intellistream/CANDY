import ctypes
import os
import sysconfig

# Automatically add the shared library path to the system path for easier imports
dll_path = os.path.join(os.path.dirname(__file__), "..", "build", "libCANDY.so")
if not os.path.exists(dll_path):
    dll_path = os.path.join(sysconfig.get_path('platlib'), "libCANDY.so")
    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"The shared library {dll_path} could not be found. Please build the project.")

# Ensure libCANDY.so is discoverable by setting LD_LIBRARY_PATH
lib_dir = os.path.dirname(dll_path)
os.environ["LD_LIBRARY_PATH"] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{lib_dir}"

# Load the shared library
vectordb = ctypes.CDLL(dll_path)
