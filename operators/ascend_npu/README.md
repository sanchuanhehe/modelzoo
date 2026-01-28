###  简介
本目录是基于Ascend NPU平台开发的算子库，支持PyTorch。

###  板端环境部署
1. 本算子库编译和运行依赖CANN包，请按照板端CANN包安装说明在板端安装CANN包
2. 请使用以下指令安装Python依赖：
```
pip3 install -r requirements.txt
```
3. 由于ONNX编译依赖protobuf-devel-3.14.0，可通过以下指令进行安装：
```
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.14.0/protoc-3.14.0-linux-aarch_64.zip
unzip protoc-3.14.0-linux-aarch_64.zip -d ./protoc
cp protoc/bin/protoc /usr/local/bin/
chmod +x /usr/local/bin/protoc
```
由protoc-3.14.0-linux-aarch_64.zip中缺失部分头文件，还需要下载源码拷贝全量头文件：
```
git clone https://github.com/protocolbuffers/protobuf.git
git checkout v3.14.0
cd protobuf
cp -rf src/google/protobuf/* /usr/include/google/protobuf/
rm /usr/include/google/protobuf/*.cc
```
至此protobuf-devel-3.14.0完成安装。

4.如果板端环境未安装gcc、g++和cmake则需进行安装（未安装或未导入CPLUS_INCLUDE_PATH环境变量会有如下报错fatal error: 'cstdint' file not found），
安装指令如下：
```
dnf install gcc g++ cmake  # 若已安装gcc和g++则不必执行此指令，执行下一条指令即可
export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/c++/12/aarch64-openEuler-linux:$CPLUS_INCLUDE_PATH"
echo 'export CPLUS_INCLUDE_PATH="/usr/include/c++/12:/usr/include/c++/12/aarch64-openEuler-linux:$CPLUS_INCLUDE_PATH"' >> ~/.bashrc
```
注意：/usr/include/c++/12 应替换为实际g++版本和安装路径

### 源码编译

#### 步骤1：编译算子库
```
cd operators/torch_npu
bash ./build.sh --soc_version=Hi3591P
```
生成的run包在`operators/torch_npu/output`目录下, 包名为`Hispark-ModelZoo-ops-xxx-linux.aarch64.run`。

#### 步骤2：安装算子run包
通过以下指令安装算子库：
```
./output/Hispark-ModelZoo-ops-0.1-linux.aarch64.run
```
默认安装到/usr/local/Ascend/latest/opp/路径下。

也可使用以下指令指定安装路径：
```
./output/Hispark-ModelZoo-ops-0.1-linux.aarch64.run --install-path=/xxx/xxx/xxx
```

#### 步骤3：添加环境变量
onnx转换om转换前需要添加环境变量：
```
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/latest/opp/vendors/customize/
export LD_LIBRARY_PATH=/usr/local/Ascend/latest/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```

若 步骤2 指定了run包安装路径则通过以下指令添加环境变量：
```
export ASCEND_CUSTOM_OPP_PATH=/xxx/xxx/xxx/vendors/customize/
export LD_LIBRARY_PATH=/xxx/xxx/xxx/opp/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
```

### 目录说明
```
├── cmake                       # cmake脚本
├── modelzoo_ops                # 自定义算子实现
│  ├── _ext_src                 # 自定义算子API适配层，将算子C++实现封装成Python模块，在线推理和onnx模型导出使用自定义算子时需要该适配层
│  ├── framework
│  │  └── op_kernel             # onnx框架适配层，将ONNX框架的算子映射成适配昇腾NPU的算子，从而完成从ONNX框架调用Ascend C自定义算子的过程
│  ├── kernel                   # 自定义算子实现
│  │  ├── CMakeLists.txt        # 算子cmake脚本
│  │  ├── cylinder_query        # cylinder_query算子实现
│  │  │  ├── op_host
│  │  │  └── op_kernel
│  └── ops                      # 算子python接口实现
├── opensource                     
│  └── ascendsamples            # 算子编译依赖脚本
├── build.sh                    # 工程编译入口
├── CMakeLists.txt              # cmake配置文件
├── README.md                   # 算子库说明
├── requirements.txt            # 算子库依赖包
└── setup.py                    # 工程编译配置
```
### 算子列表(当前仅支持模型推理)
```
cylinder_query
furthest_point_sampling
group_points
```
#### PyTorch版本
- 支持PyTorch 2.1.0

#### 硬件支持
支持型号：Hi3591P