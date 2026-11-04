import argparse
import json
from .agent import run_agent, run_direct
from .tools import get_wifi_metrics


def main():
    parser = argparse.ArgumentParser(description="校园 Wi-Fi 慢速排查 Agent")
    parser.add_argument("question", help="问题文字，包含 AP 编号")
    parser.add_argument("--mode", choices=("agent", "direct"), default="agent")
    parser.add_argument("--ap", help="direct 模式读取同一 AP 快照的编号")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        if args.mode == "direct":
            if not args.ap:
                parser.error("direct 模式需要 --ap")
            result = run_direct(args.question, get_wifi_metrics(args.ap))
        else:
            result = run_agent(args.question)
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, f"错误：{exc}\n")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("问题：", args.question)
        for index, step in enumerate(result["trace"], 1):
            print(f"\n工具调用 {index}：{step['tool']}")
            print("输入：", json.dumps(step["input"], ensure_ascii=False))
            print("返回：", json.dumps(step["output"], ensure_ascii=False))
        print("\nAgent 结论：\n" + result["answer"])


if __name__ == "__main__":
    main()
