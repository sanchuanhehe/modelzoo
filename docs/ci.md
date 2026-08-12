# GitHub Actions CI

本仓库的 GitHub Actions 分成不依赖板卡的标准 Runner 流水线和后续独立接入的 HIL 流水线。当前工作流不包含 HIL，也不会连接开发板。

## 已启用的检查

- `CI / Repository validation`：JSON 和 `build_config.json` 结构、Python 语法、Shell 语法和静态检查。
- `CI / Toolchain configuration`：检查标准 Runner 交叉编译所需的六项 SDK 配置；缺少时明确跳过交叉编译，不会让基础 CI 无故变红。
- `CI / Sample build / svp-nnn`：使用标准 Ubuntu Runner 交叉编译 ResNet50 的 SVP_NNN 配置。
- `CI / Sample build / nnn`：使用标准 Ubuntu Runner 交叉编译 ResNet50 的 NNN 配置。
- `CodeQL`：以无构建模式分析 C/C++ 和 Python。
- `Dependency review`：PR 引入高危依赖时失败。
- Dependabot：每周检查 GitHub Actions 版本。

交叉编译以两个代表性构建作为快速门禁。仓库完整的 91 条构建配置由元数据检查覆盖；后续可根据构建耗时把全部样例拆成定时矩阵。

## GitHub 配置

在 `Settings -> Secrets and variables -> Actions` 配置以下值。

### Repository variables

| 名称 | 内容 |
| --- | --- |
| `MODELZOO_SVP_NNN_CANN_SHA256` | 原厂 SVP_NNN PC 归档的小写 SHA-256。 |
| `MODELZOO_NNN_CANN_SHA256` | 原厂 NNN PC 归档的小写 SHA-256。 |
| `MODELZOO_TOOLCHAIN_SHA256` | 原厂 `aarch64-mix210-linux.tgz` 的小写 SHA-256。 |

公开 SVP_NNN 安装包的原始下载地址及其 SHA-256 为：

```text
SOURCE_URL=https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz
SOURCE_SHA256=bc299b05b20b583f50ad8695502f7f1e30b4cdbd4635b0ba63e19849b071b314
```

流水线可以直接处理原厂 PC 归档：先展开外层 `.tgz`，再仅安装目标侧 `acllib/runtime --devel` 包。它不会安装 ATC、主机侧编译器或驱动，也不要求自托管 Runner。流水线同时兼容已安装式可重定位归档。

原厂 GCC 7.3 前端依赖 `libisl.so.19`。Ubuntu 24.04 已不再提供该 ABI；配置脚本会从 Ubuntu 18.04 官方归档下载固定 SHA-256 的 `libisl19` 包，只解到 Runner 临时目录，并通过 `LD_LIBRARY_PATH` 供编译器使用。

SVP_NNN 和 NNN 必须分别配置自己的归档，不能让两个引擎误用同一 CANN 包。

### Repository secrets

| 名称 | 内容 |
| --- | --- |
| `MODELZOO_SVP_NNN_CANN_URL` | GitHub Runner 可下载的原厂 SVP_NNN PC 归档地址。 |
| `MODELZOO_NNN_CANN_URL` | GitHub Runner 可下载的原厂 NNN PC 归档地址。 |
| `MODELZOO_TOOLCHAIN_URL` | GitHub Runner 可下载的原厂 `aarch64-mix210-linux.tgz` 地址。 |

三个 URL 按 Secret 管理，是因为这些包通常需要从 SDK/FAE 渠道获取，而且下载地址可能包含授权信息。不要把 GitHub PAT 或云存储长期密钥放进 URL；优先使用只读、最小权限的下载地址。脚本会在使用前校验 Repository variable 中对应的 SHA-256。

默认不把原厂 SDK 或工具链写入公开仓库的 GitHub Actions cache。若后续确认相关包允许在该仓库的 Actions 范围内分发，可再按引擎和 SHA-256 增加缓存；在此之前由私有下载端控制访问权限和缓存策略。

Fork PR 不运行需要这些凭据的交叉编译任务，但仍运行无密钥的仓库验证。GitHub 默认也不会向 Fork PR 下发仓库 Secrets。

## 分支保护

第一次成功运行后，在 `Settings -> Rules -> Rulesets` 中保护 `master`，建议将下列检查设为必需：

- `Repository validation`
- `Sample build / svp-nnn`
- `Sample build / nnn`
- `Analyze cpp`
- `Analyze python`
- `Dependency review`

在交叉编译的六个配置值完整设置以前，不要将两个 Sample build 检查设为必需；它们会按设计跳过。可先把 `Repository validation`、`Toolchain configuration`、CodeQL 和 `Dependency review` 设为必需检查。

## 本地验证

```bash
python3 ci/validate_repository.py
python3 ci/check_python_sources.py
bash -n ci/*.sh
shellcheck ci/*.sh samples/*.sh utils/*.sh
```
