import sys
print("Python sys.path:")
for path in sys.path:
    print(path)

from pyvectordb import VectorDB
from pyvectordb import VectorDBStream