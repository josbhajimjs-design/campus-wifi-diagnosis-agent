# 校园 Wi-Fi 慢速排查 Agent

作者：谢鑫强（23211053），单人完成。课程：《人工智能的网络应用》。

GitHub 仓库：https://github.com/josbhajimjs-design/campus-wifi-diagnosis-agent

本项目用本地 Ollama 模型选择并调用两个只读 Tool，分析校园 Wi-Fi 慢速或状态问题。`data/snapshots.json` 全部是**为教学实验构造的模拟快照**，不代表真实校园网络；程序不会扫描或改动设备。

## 环境与模型

- Windows、macOS 或 Linux；Python 3.10 或更高版本。Python 代码只使用标准库，`requirements.txt` 无 pip 依赖。
- 本地安装 Ollama，并执行 `ollama pull qwen3:4b`、`ollama serve`。本次实验为 Ollama 0.40.6、`qwen3:4b`，温度 0。
- 如模型或地址不同，设置 `OLLAMA_MODEL`、`OLLAMA_HOST` 环境变量；默认分别为 `qwen3:4b`、`127.0.0.1:11434`。
- 在本 README 所在目录运行命令。模型首次加载可能较慢，单次请求超时为 180 秒。

## 完整操作

```powershell
python -m src.main '宿舍 AP-DORM-02 信号很好但下载只有 11 Mbps，为什么慢？'
```

输出包含用户问题、实际工具调用的参数与返回值、模型结论。也可加 `--json` 保存结构化日志。

```powershell
python -m src.main '宿舍 AP-DORM-02 信号很好但下载只有 11 Mbps，为什么慢？' --mode direct --ap AP-DORM-02
```

`direct` 模式把与 Agent 查询相同的模拟快照直接写进提示词，不开放工具；因此两组有相同的事实输入。模型直接回答不执行程序计算。

## 测试和复现

```powershell
python -m tests.self_check
python -m tests.evaluate
```

第一条验证香农公式、缺失值、双高占用和非法范围；第二条重跑六个案例的 Agent/直接 LLM 对照，把原始回答、工具日志和耗时逐例写到 `results/raw_runs.json`。`tests/cases.json` 给出输入、合理判断标准和测试目标。模型生成有随机性，即使温度为零也可能因运行版本变化产生差异；请以本次保存的原始日志为报告结果依据。

## 设计与限制

- `get_wifi_metrics(ap_id)` 只查模拟快照；`assess_wifi_link(...)` 用 `C=B log₂(1+10^(SNR_dB/10))` 计算理想物理层上界，并返回原始占用率。上界不是应用吞吐量。
- Agent 利用 Ollama `/api/chat` 的工具调用机制选择 Tool。调度层把 AP ID 与用户输入核对，并把计算参数锁定为查询所得数据，避免小模型抄错或编造参数；日志同时保留模型请求参数和实际执行参数。模型根据工具结果撰写最终解释；证据检查发现过度断言时最多要求模型重写两次，仍未通过才使用保守回退结论。`model_draft` 和 `answer_source` 展示这条路径。
- 高信道忙碌率、高上联利用率只支持排查假设。单次快照无法证明根因；缺失两个关键占用率时尤其应承认不确定。
- 本实验没有真实设备的连续采样、频段/信道配置、重传率、端到端丢包数据；对现场问题需要进一步测量。模型仍可能错误解读指标，人工应复核结论。

## 文件

`src/agent.py` 为 Agent 和 Ollama 调用，`src/tools.py` 为两个 Tool，`src/main.py` 为命令行入口，`data/` 为模拟快照，`tests/` 为案例和可运行检查，`results/` 为实际实验记录，`docs/` 为报告图与运行截图，`report.pdf` 为课程报告。
