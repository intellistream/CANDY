import sys
print("Python sys.path:")
for path in sys.path:
    print(path)

from pyvectordb import VectorDB

VectorDB.insert_vector()
# from pyvectordb import VectorDBStream