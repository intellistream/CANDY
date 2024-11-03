# CANDY Python Package Builder

This folder packages `.so` files into a Python `.whl` file, allowing easy installation into virtual environments via `pip install`.

## Prerequisites
- Build the CANDY library, ensuring `.so` files are generated and available in the Python site-packages directory or the specified build directory.

## Packaging Instructions

1. **Copy `.so` files to `./lib/` directory** in this folder. Verify that the `.so` files are present in the `site-packages` directory, or locate them in the designated build path.

    ```bash
    mkdir -p ./lib
    cp $(python3 -c "import site; print(site.getusersitepackages())")/*.so ./lib/
    ```

2. **Build the Python package** (creates a `.whl` file):

    ```bash
    python setup.py sdist bdist_wheel
    ```

3. **Install the package** into your virtual environment:

    ```bash
    pip install dist/candy-0.1.0-py3-none-any.whl
    ```

4. **Alternative Installation via `requirements.txt`**:
   Add the `.whl` file path to `requirements.txt` for easy setup in any environment:

    ```
    /path/to/candy-0.1.0-py3-none-any.whl
    ```

5. **Try candy by** using `import candy.pycandy` in python.