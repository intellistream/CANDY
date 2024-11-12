from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import subprocess

# Custom command to set LD_LIBRARY_PATH to include PyTorch libraries
class CustomInstallCommand(install):
    def run(self):
        try:
            # Fetch the PyTorch library path
            torch_lib_path = subprocess.check_output(
                "python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), \"lib\"))'",
                shell=True
            ).decode().strip()

            # Add torch library path to LD_LIBRARY_PATH
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{os.getenv('LD_LIBRARY_PATH', '')}"
            print(f"LD_LIBRARY_PATH set to include PyTorch libraries: {torch_lib_path}")

        except Exception as e:
            print(f"Error setting LD_LIBRARY_PATH for PyTorch: {e}")

        # Proceed with the standard installation
        super().run()

if __name__ == "__main__":
    setup(
        name="candy",
        version="0.1.0",
        description="A Python package that includes compiled C++ extensions of CANDY.",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        author="IntelliStream",
        author_email="your_email@example.com",
        packages=find_packages(),
        package_data={
            "candy": ["*.so"]  # Include all .so files in the candy package
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6",
        install_requires=[
            "torch==2.4.0"  # Specify PyTorch version 2.4.0
        ],
        zip_safe=False,
        cmdclass={
            "install": CustomInstallCommand  # Use the custom install command
        },
    )
