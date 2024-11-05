import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import glob
import torch

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        # Check if CMake is installed
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # Set environment variables
        os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
        if sys.platform == 'linux':
            os.environ['LD_LIBRARY_PATH'] = '/path/to/custom/libs:' + os.environ.get('LD_LIBRARY_PATH', '')

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        build_path = "build"
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        torchCmake = torch.utils.cmake_prefix_path
        threads = str(os.cpu_count())
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                    '-DCMAKE_PREFIX_PATH='+torchCmake,
                   ]
        
        cfg = 'Debug' if self.debug else 'Release'
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args = ['--config', cfg]
        build_args +=  ['--', '-j'+threads]
        subprocess.run(['cmake', ext.sourcedir] + cmake_args,cwd= build_path,check=True)
        subprocess.run(['cmake', '--build', '.'] + build_args,cwd= build_path,check=True)
        # cwd

        # Now copy all *.so files from the build directory to the final installation directory
        so_files = glob.glob(os.path.join(build_path, '*.so'))
        for file in so_files:
            shutil.copy(file, extdir)
if  __name__ == "__main__":
    setup(
        name='PyCANDYFramework',
        version='0.0.0',
        author='IntelliStream',
        description='A simple python version of CANDY Framework built with Pybind11 and CMake',
        long_description='',
        ext_modules=[CMakeExtension('.')],
        cmdclass={
            'build_ext': CMakeBuild,
        },
        zip_safe=False,
    )