# 实验记录与验证改进

本轮修改加强记录保护和报告数据完整性，保留既有算法、奖励、预算、历史模型和结果。没有重新选择成功种子，也没有用新实验替换报告数据。

## 实现范围

| 改进 | 修改位置 | 验收行为 |
|---|---|---|
| 扫描及适应实验保护 | run_sweep.py、run_adaptation.py、provenance.py | 普通运行拒绝非空目录；显式恢复检查配置和源码 |
| 扫描来源留档 | run_sweep.py、verify_results.py | 新扫描保存并验证源码快照和环境；恢复保留原开始时间 |
| 正式分析完整性 | analysis.py、provenance.py | 缺种子、多种子、缺模型或日志时拒绝汇总 |
| 随机基线完整性 | analysis.py、verify_results.py | 缺失、空数据、重复或缺少 seed/episode 时失败 |
| 冻结评价完整性 | verify_results.py | 每个预期 method/seed 恰好出现一次 |
| 报告图表输入检查 | tools/export_report_figures.py | 检查主实验和改图实验的模型与记录目录完整性 |
| 贪婪评价序列 | analysis.py、verify_results.py、tools/export_report_figures.py | 主实验、扫描、改图评价须符合间隔和最后一回合；缺失、重复、错序、错种子或累计步数不符即失败 |
| 探索率边界 | configuration.py、run_sweep.py、verify_results.py | 零下限不再触发 log(0)，已在下限时显示 0；不可有限次达到的下限以空 CSV 字段表示 |
| 没观察到收敛的扫描 | verify_results.py | 空收敛统计按未观察到处理，不尝试转换为空字符串浮点数 |

来源检查使用源码哈希；修改源码后应另开结果目录。原始扫描缺少运行级 metadata，允许核对已有策略和统计，但验证会明确报告未核验来源哈希。仅有历史 source_snapshot 文件夹不能证明当时记录了哪些源码哈希。

## 检查命令

在仓库根目录安装 requirements-lock.txt，然后执行：

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
.venv/bin/python verify_results.py --results results/main --sweep results/sweep
.venv/bin/python verify_results.py --results results/adaptation --adaptation results/adaptation --base-results results/main
```

验证脚本会写入验证输出文件；它不训练、不改变最终模型。若要保持历史目录每个字节不变，可复制结果到临时目录后验证。

tests/test_final_submission.py 进一步覆盖完整评价序列、零探索下限的真实扫描和验证、图表输入异常拒绝，以及 Pygame 的手动移动、暂停、恢复、重置和退出事件。

新增 tests/test_workflows.py 覆盖异常数据拒绝、两个入口防覆盖、新扫描来源核验，以及扫描和适应实验恢复时保留已完成运行。已有恢复测试继续验证回合边界恢复与连续训练一致。

## 与 Part E 模板的对应

此表是代码证据导航，用于你自行完成报告，不代替报告正文。模板来源为 Project-FinalReport-Template-Part-E (1).docx。

| 模板栏目 | 可以核查的项目材料 |
|---|---|
| 1.4 Dataset/Environment | config.json、warehouse/env.py；交互生成经验，没有外部训练数据集 |
| 2.1 工作流程 | docs/architecture.md 的系统流程和训练入口 |
| 2.2 系统架构 | warehouse 各模块职责、agents.py 的两种方法、各 run_*.py 的实验协议 |
| 2.3 GUI/Environment | demo.py 和 results 内保存的 GUI 截图；演示时核查键盘操作 |
| 3.1 Experimental Settings | 各结果目录 config.json、metadata.json；区分主训练和适应额外预算 |
| 3.2 Pre-Processing | env.py 的状态编码和坐标归一化；无外部数据清洗流程 |
| 3.3 Experimental Results | results 的原始 CSV、summary、final_policies 和 output/figures |
| 3.4 Limitations | 单个固定主图、一个改图；DQN 失败种子；缺少从零训练对照 |
| 附录 A、B | 每人真实贡献、团队商定比例和本人签名，由团队填写 |

从零训练对照与 DQN 独立调参是后续研究增强项，本轮未实施，也不把它们写成已完成结果。
