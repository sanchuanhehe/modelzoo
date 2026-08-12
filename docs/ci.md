# GitHub Actions CI

本仓库的 pre-HIL 流水线全部运行在 GitHub 标准 `ubuntu-24.04` Runner。它不连接板卡、不使用自托管 Runner，也不执行刷机、串口或 SSH。

## 固定 SDK 事实源

CI 只读取 [ci/sdk-lock.json](../ci/sdk-lock.json)：其中固定 GitHub 仓库、Release tag、附件名、字节数、SHA-256 和 SVP_NNN 分片顺序。工作流不使用 `/latest`，也不需要 Repository Secrets/Variables。

当前 Release：`sdk-ss928v100-r001c02spc022-ci.1`。发布附件包括：

- 原厂 `aarch64-mix210-linux.tgz`；
- 原厂 `NNN_PC.tgz`；
- 原厂 `SVP_NNN_PC_V1.0.2.17.tgz` 的两个原始字节分片；
- `sdk-manifest.json`、`SHA256SUMS`、来源说明和再分发说明。

下载脚本逐附件检查 size/SHA，SVP_NNN 按锁定顺序重组并再次检查原始归档 size/SHA。工具链内层 `tar.bz2` 会自动展开；`cc1/cc1plus` 缺少的 `libisl.so.19` 从固定 SHA 的 Ubuntu 18.04 官方包解到 Runner 临时目录，不替换系统库。

## 工作流

- `CI / Repository validation`：JSON/build_config、sdk-lock schema、Python、Ruff、ShellCheck、License/OAT、大文件和 SDK/模型禁入 Git。
- `CI / Select affected targets`：模型目录变更只选择相关目标；common、工具链、lock 或 workflow 变更选择双引擎 ResNet50 代表矩阵；纯文档变更不下载 SDK。
- `CI / Sample build / ...`：内部 PR、master push 与手动运行使用标准 Runner 真实交叉编译，并验证 `ARM aarch64` ELF。
- `CI / Model conversion / SVP_NNN`：生成固定 SHA 的可再分发 ONNX 和校准输入，使用锁定 CANN/ATC 转换为 OM。
- `Nightly full build`：按引擎分成两个 job；每个 job 只安装一次 SDK，然后顺序构建去重后的全部 SS928 目标并输出逐目标日志/汇总。当前支持 79 项，非 SS928 的 12 项明确跳过。
- `CodeQL`：C/C++ 与 Python 静态分析。
- `Dependency review`：PR 新增高危依赖时失败。

外部 Fork PR 只运行无 SDK 门禁，避免数 GiB 下载被滥用；它不会获得任何凭据，因为 pre-HIL 构建本身也不需要 Secrets。

## 构建与 HIL 输入产物

编译工作流上传 `main`、`build.log`、`expected.json`、`build-manifest.json` 和 `SHA256SUMS`。转换工作流上传：

```text
model.om
input/model.onnx
input/input.bin
expected.json
atc.log
conversion-manifest.json
SHA256SUMS
```

manifest 记录 commit、SDK Release、引擎、SOC、CMake/ATC 参数和每个输出 SHA。`expected.json` 明确标记 `not-run`；产物只是将来 HIL Runner 的可追溯输入，不表示已在板卡运行。

## 本地门禁

```bash
python3 ci/validate_repository.py
python3 ci/validate_sdk_lock.py
python3 ci/check_repository_policy.py
python3 ci/check_license_policy.py
python3 ci/check_python_sources.py
bash -n ci/*.sh ci/sdk-release/*.sh
shellcheck ci/*.sh ci/sdk-release/*.sh
```

## 分支保护

`master` 应要求：

- `Repository validation`
- `Analyze cpp`
- `Analyze python`
- `Dependency review`
- `Cross-build gate`
- `Pre-HIL artifact gate`

内部 PR 和 `master` 构建会把代表性 ResNet50 的 AArch64 `main`、转换后的
`model.om`、固定输入、期望状态和完整哈希清单组装为单一 pre-HIL 工件。该步骤只
准备板卡测试输入，不连接 Runner 控制机或板卡，也不执行刷机、串口或 SSH。

夜间工作流按 NNN 和 SVP_NNN 分成两个标准 Runner job，每个 job 只安装一次对应
SDK，再顺序构建该引擎的全部 SS928 目标。最终 `Nightly summary` 会报告 scheduled、
success、failed 和 skipped 数量，并保留逐目标日志与 TSV 清单。
- `Model conversion / SVP_NNN`

夜间全量构建用于回归覆盖，不阻塞每个 PR。HIL 检查在自托管 Runner 真正接入后另行增加，不能用 pre-HIL 构建替代。
