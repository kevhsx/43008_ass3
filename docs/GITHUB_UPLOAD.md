# GitHub 上传说明

## 目录内容

- `warehouse/`：环境、算法、训练、分析和恢复模块。
- 根目录 Python 脚本、配置及依赖清单：运行和复现实验。
- `tests/`：自动化测试。
- `results/main/`、`results/sweep/`、`results/adaptation/`：报告依据，包括原始日志、最终模型、配置及历史源码快照。
- `output/jupyter-notebook/`：已执行分析笔记本。
- `output/figures/`：报告图表。
- `scripts/macos/`：可选双击启动器，已调整为从仓库根目录运行。
- `docs/`：架构说明、实验记录指南、上传步骤和历史验证记录。

原始项目未移动或修改。此副本省略虚拟环境、缓存、编辑器配置、训练恢复 checkpoint 和内部审查文档。历史源码快照保持原样，不应手动清理或修改。

## 上传

在 GitHub 创建空仓库。课程项目可先选择 Private，并按课程要求向教师或组员开放访问。此副本已初始化本地 Git，但未创建提交，也未连接远程仓库。

在本目录终端运行（将 REMOTE_URL 替换为实际仓库地址）：

```bash
git status --short
git add .
git diff --cached --stat
git commit -m "Prepare warehouse RL project and experiment records"
git remote add origin REMOTE_URL
git push -u origin main
```

若通过网页上传，上传本文件夹内的项目内容。优先使用 Git，以包含 `.gitignore` 和 `.gitattributes` 等隐藏文件。不要上传 `.git/`、`.venv/` 或整个项目压缩包作为源码替代。

本次整理未指定开源许可证。若需要公开和许可他人复用，请由团队确认授权与课程规定后选择许可证。

## 下载后的使用

使用 Python 3.12，在仓库根目录按 README 创建独立虚拟环境并安装依赖。原目录的虚拟环境不会随仓库上传。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python demo.py --method q_learning --seed 0
.venv/bin/python verify_results.py --results results/main --sweep results/sweep
.venv/bin/python verify_results.py --results results/adaptation --adaptation results/adaptation --base-results results/main
```

Windows 命令见 README。

## 实验记录保护

`results/main`、`results/sweep` 和 `results/adaptation` 是固定的报告证据。新实验必须写入全新目录。参数扫描与适应实验现已接入非空目录保护；恢复必须显式使用 `--resume`，并保持配置和源码一致。工程修复没有更换历史模型或实验结论。

省略的 `checkpoint.zip` 是训练恢复状态；最终 `model.pt` 和 `q_table.npy` 已保留，可以演示及核对结果，但不能依靠它们完整恢复历史训练过程。

历史元数据包含原始机器路径，用于保留实验来源；适应实验验证时使用上面的 `--base-results`。公开前请检查团队是否愿意公开这些来源信息、笔记本输出及项目署名。

## 整理后的导航

目录重要程度见根目录 README；模块职责和修改位置见 [架构说明](architecture.md)；实验文件保留规则见 [实验记录指南](EXPERIMENT_RECORDS.md)。`validation/` 中是历史验证输出，不代表本次重新执行的检查。
