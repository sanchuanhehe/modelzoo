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
| `MODELZOO_SVP_NNN_CANN_URL` | 已安装式 SVP_NNN CANN 归档下载地址。解压后必须包含 `acllib/include/acl` 和 `acllib/lib64/stub`。 |
| `MODELZOO_SVP_NNN_CANN_SHA256` | 上述 SVP_NNN 归档的小写 SHA-256。 |
| `MODELZOO_NNN_CANN_URL` | 已安装式 NNN CANN 归档下载地址。 |
| `MODELZOO_NNN_CANN_SHA256` | 上述 NNN 归档的小写 SHA-256。 |

公开 SVP_NNN 安装包的原始下载地址及其 SHA-256 为：

```text
SOURCE_URL=https://hispark-obs.obs.cn-east-3.myhuaweicloud.com/SVP_NNN_PC_V1.0.6.0.tgz
SOURCE_SHA256=bc299b05b20b583f50ad8695502f7f1e30b4cdbd4635b0ba63e19849b071b314
```

该原始包内部仍包含 `.run` 安装器，不能直接作为 `MODELZOO_SVP_NNN_CANN_URL`。应先在授权环境中安装，再把安装目录打成可重定位 tar 归档并重新计算 SHA-256，供无特权的 GitHub Runner 解压使用。

SVP_NNN 和 NNN 必须分别配置自己的归档，不能让两个引擎误用同一 CANN 包。

### Repository secrets

| 名称 | 内容 |
| --- | --- |
| `MODELZOO_TOOLCHAIN_URL` | 可由 GitHub Runner 下载的 `aarch64-mix210-linux` 可重定位 tar 归档地址。 |
| `MODELZOO_TOOLCHAIN_SHA256` | 工具链归档的小写 SHA-256。 |

工具链 URL 按 Secret 管理，是因为仓库文档说明该工具链通常需要从 SDK/FAE 渠道获取。归档解压后必须包含 `aarch64-mix210-linux-gcc`。不要把 GitHub PAT 或云存储长期密钥放进 URL；优先使用只读、短权限或仓库可访问的下载地址。

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
