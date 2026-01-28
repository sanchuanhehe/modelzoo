# Copyright (c) ModelZoo. 2025-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import subprocess
import multiprocessing
import torch

from setuptools._distutils.version import LooseVersion
from setuptools.command.build_ext import build_ext
from setuptools.command.build_clib import build_clib
from setuptools import Extension
from setuptools import find_packages, setup

PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

class CLibBuildConfig:
    def __init__(self, build_temp, kernel_name):
        self.build_temp = build_temp
        self.kernel_name = kernel_name
    
    def get_cmake_args(self, modelzoo_ops_dir):
        """ 生成CMake参数列表 """
        args = [
            "-B",
            self.build_temp,
            f"-DMODELZOO_OPS_PATH={modelzoo_ops_dir}",
            f"-DKERNEL_NAME={self.kernel_name if self.kernel_name else '*'}",
        ]
        return args

    def get_build_args(self):
        """ 生成build参数列表 """
        build_args = ["--build", self.build_temp, f"-j{multiprocessing.cpu_count()}"]
        return build_args

class OpCLibBuild(build_clib):
    def initialize_options(self) -> None:
        super().initialize_options()
        self.kernel_name = None

    def run(self) -> None:
        self.cmake = "cmake"
        config = CLibBuildConfig(self.build_temp, self.kernel_name)

        cmd = self.get_finalized_command("build_py")
        modelzoo_ops_dir = os.path.join(PROJECT_DIR, cmd.build_lib, cmd.get_package_dir("modelzoo_ops"))
        if not os.path.exists(modelzoo_ops_dir):
            os.makedirs(modelzoo_ops_dir)

        cmake_args = config.get_cmake_args(modelzoo_ops_dir)
        build_args = config.get_build_args()

        for step in range(0, 2):
            subprocess.check_call(
                [self.cmake, PROJECT_DIR] + cmake_args + ["-DCMAKE_STEP=" + str(step)],
                cwd=PROJECT_DIR,
                env=os.environ,
            )
            subprocess.check_call(
                [self.cmake] + build_args,
                cwd=PROJECT_DIR,
                env=os.environ,
            )

class ExtBuildConfig:
    def __init__(self, build_temp):
        self.build_temp = build_temp
        self.python_executable = sys.executable
        
        # 获取Torch配置
        self.torch_version = torch.__version__
        self.use_xla = LooseVersion(self.torch_version) < LooseVersion("2.1.0")
        self.torch_abi = 1 if torch.compiled_with_cxx11_abi() else 0
        
        # 编译器标志
        self.cxx_flags = self.get_cxx_flags()

    def get_cxx_flags(self):
        """ 获取C++编译器标志 """
        flags = ["-std=c++17"]
        
        # 添加PyBind11相关宏定义
        for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
            val = getattr(torch._C, f"_PYBIND11_{name}", None)
            if val:
                flags.append(f"-D_PYBIND11_{name}={val}")
        
        return flags
    
    def get_cmake_args(self, modelzoo_ops_dir):
        """ 生成CMake参数列表 """
        args = [
            "-B",
            self.build_temp,
            f"-DMODELZOO_OPS_PATH={modelzoo_ops_dir}",
            f"-DEXT_CXX_FLAGS={' '.join(self.cxx_flags)}",
            f"-DPython3_EXECUTABLE={self.python_executable}",
        ]
        if self.use_xla:
            args.append("-DCOMPILE_WITH_XLA:BOOL=ON")

        if self.torch_abi:
            args.append("-DABI=1")
        else:
            args.append("-DABI=0")
        return args

    def get_build_args(self):
        """ 生成build参数列表 """
        build_args = ["--build", self.build_temp, f"-j{multiprocessing.cpu_count()}"]
        return build_args

class OpExtBuild(build_ext):
    def run(self) -> None:
        self.cmake = "cmake"
        config = ExtBuildConfig(self.build_temp)
        
        cmd = self.get_finalized_command("build_py")
        modelzoo_ops_dir = os.path.join(PROJECT_DIR, cmd.build_lib, cmd.get_package_dir("modelzoo_ops"))
        if not os.path.exists(modelzoo_ops_dir):
            os.makedirs(modelzoo_ops_dir)

        cmake_args = config.get_cmake_args(modelzoo_ops_dir)
        build_args = config.get_build_args()

        for step in range(2, 4):
            subprocess.check_call(
                [self.cmake, PROJECT_DIR] + cmake_args + ["-DCMAKE_STEP=" + str(step)],
                cwd=PROJECT_DIR,
                env=os.environ,
            )
            if step == 3:
                build_args = ["--build", self.build_temp, f"-j{multiprocessing.cpu_count()}", "--target package"]
            subprocess.check_call(
                [self.cmake] + build_args,
                cwd=PROJECT_DIR,
                env=os.environ,
            )

setup(
    name="modelzoo_ops",
    description="Hispark ModelZoo Operators Library on Ascend-NPU.",
    keywords="modelzoo_ops",
    ext_modules=[Extension("modelzoo_ops._ext", sources=[])],
    libraries=[("modelzoo_ops", {"sources": []})],
    cmdclass={
        "build_clib": OpCLibBuild,
        "build_ext": OpExtBuild,
    },
    packages=find_packages(),
    include_package_data=True,
)
