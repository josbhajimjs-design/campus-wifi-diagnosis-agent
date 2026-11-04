"""Small Ollama tool-calling agent; all measurements come from tools."""
import json
import os
import re
import urllib.error
import urllib.request
from .tools import assess_wifi_link, get_wifi_metrics


MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434").rstrip("/")
if not HOST.startswith(("http://", "https://")):
    HOST = "http://" + HOST
SYSTEM = ("/no_think 你是校园 Wi-Fi 排障助手。先识别用户给出的完整 AP 编号。排障时先调用 get_wifi_metrics 查测量值，"
          "再根据情况调用 assess_wifi_link 计算物理层理论上界并查看资源占用。不要编造数据。"
          "分别考虑信号质量、无线争用、上联拥塞；观测只能支持或削弱假设，不能证明根因。"
          "缺测时明确说证据不足。最终用中文写出：现象、工具调用、观测与计算、可能原因、结论、下一步。"
          "香农上界不是实际吞吐量。")

TOOLS = [
    {"type": "function", "function": {"name": "get_wifi_metrics", "description": "按 AP 编号查询模拟测量快照，不返回诊断结论", "parameters": {"type": "object", "properties": {"ap_id": {"type": "string"}}, "required": ["ap_id"]}}},
    {"type": "function", "function": {"name": "assess_wifi_link", "description": "根据 SNR、信道带宽、信道忙碌率和上联利用率计算香农上界并整理资源占用；缺测值用 null", "parameters": {"type": "object", "properties": {"snr_db": {"type": "number"}, "bandwidth_mhz": {"type": "number"}, "channel_busy_pct": {"type": ["number", "null"]}, "backhaul_util_pct": {"type": ["number", "null"]}}, "required": ["snr_db", "bandwidth_mhz", "channel_busy_pct", "backhaul_util_pct"]}}},
]
FUNCTIONS = {"get_wifi_metrics": get_wifi_metrics, "assess_wifi_link": assess_wifi_link}
SYNTHESIS = ("你是校园 Wi-Fi 排障 Agent。只依据实际工具结果，用中文回答六行：现象、工具、证据、可能原因、结论、下一步。"
             "这些是一次模拟快照，任何指标都只能支持排查方向，不能证明根因。缺测既不支持也不反驳假设。"
             "香农容量是理想物理层上界，不等于应用吞吐量；信道忙碌率不是冲突率。"
             "若 SNR 很低且 RSSI 很弱，优先查信号；若信道忙碌率很高，优先查无线占用；"
             "若上联利用率很高，优先查上联；两项都高则同时排查，不可只断定一种；两项缺失则无法判断是否拥塞。"
             "不得把客户端数量直接等同于拥塞。禁止使用‘导致’‘主因’‘已证明’‘假设不成立’‘排除’‘非瓶颈’等确定措辞。"
             "结论用‘优先排查’‘不能确定’‘当前未见明显瓶颈’。最多 220 汉字。")


def chat(messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 1800) -> dict:
    body = {"model": MODEL, "messages": messages, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": max_tokens}}
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(HOST + "/api/chat", json.dumps(body).encode("utf-8"),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)["message"]
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"无法连接本地 Ollama：{exc}") from exc


def grounded_answer(trace: list[dict]) -> str:
    """Constrain the final conclusion to what one snapshot can support."""
    snapshot = next((x["output"] for x in trace if x["tool"] == "get_wifi_metrics" and "error" not in x["output"]), None)
    if not snapshot:
        return "未取得有效测量数据，无法判断。请核对 AP 编号。"
    result = next((x["output"] for x in trace if x["tool"] == "assess_wifi_link" and "error" not in x["output"]), None)
    ap = snapshot["ap_id"]
    basis = f"RSSI {snapshot['rssi_dbm']} dBm、SNR {snapshot['snr_db']} dB、实测速率 {snapshot['throughput_mbps']} Mbps"
    if not result:
        return f"现象：{ap} 状态查询。\n工具：get_wifi_metrics。\n证据：{basis}。\n结论：已返回单次模拟快照；尚未评估链路瓶颈。\n下一步：若遇到慢速，再检查信道和上联占用。"
    busy, backhaul = result["channel_busy_pct"], result["backhaul_util_pct"]
    basis += f"；信道忙碌率 {str(busy) + '%' if busy is not None else '缺失'}、上联利用率 {str(backhaul) + '%' if backhaul is not None else '缺失'}"
    basis += f"；香农理想上界 {result['shannon_upper_bound_mbps']} Mbps（非实测速率）"
    # ponytail: these teaching thresholds are heuristic; replace with device baselines and repeated samples for real deployment.
    if busy is None or backhaul is None:
        cause = "无线争用或上联瓶颈均未排除"
        conclusion = "关键占用率缺失，不能确定是否拥塞或判断根因"
        action = "补测信道忙碌率、上联利用率，并复测端到端吞吐与丢包"
    elif snapshot["snr_db"] <= 10 and snapshot["rssi_dbm"] <= -80:
        cause = "低 SNR 与弱 RSSI 支持信号质量问题；当前信道和上联占用较低，削弱拥塞解释"
        conclusion = "优先排查信号质量，单次快照不能证明根因"
        action = "在客户端位置复测信号、噪声和重传率"
    elif busy >= 80 and backhaul >= 90:
        cause = "无线信道与上联均有高占用，两种瓶颈假设同时得到支持"
        conclusion = "不能仅凭一次快照确定无线侧或上联侧的唯一根因"
        action = "同步采集无线重传、上联丢包和分时流量，再做隔离测试"
    elif busy >= 80:
        cause = "高信道忙碌率支持无线资源争用；客户端多也值得检查，上联高负载证据较弱"
        conclusion = "优先排查无线占用，不能据此证明碰撞或根因"
        action = "核查各客户端流量、邻近信道占用和重传率"
    elif backhaul >= 90:
        cause = "高上联利用率支持上联瓶颈；无线占用和信号质量证据较弱"
        conclusion = "优先排查上联，尚不能证明其为根因"
        action = "核查上联流量峰值、接口错误及丢包"
    else:
        cause = "当前快照未见明显弱信号、无线高占用或上联高负载"
        conclusion = "当前状态未见明显瓶颈，不代表其他时段均正常"
        action = "若用户仍感到慢，采集问题时段的连续测量及客户端数据"
    return f"现象：{ap} 网络状态调查。\n工具：get_wifi_metrics、assess_wifi_link。\n证据：{basis}。\n可能原因：{cause}。\n结论：{conclusion}。\n下一步：{action}。"


def review_answer(answer: str, trace: list[dict]) -> str | None:
    """Return one concrete issue for the model to repair, or None if supported."""
    snapshot = next((x["output"] for x in trace if x["tool"] == "get_wifi_metrics" and "error" not in x["output"]), None)
    if not snapshot:
        return "没有有效测量值"
    if len(answer) > 500 or not all(label in answer for label in ("现象：", "证据：", "可能原因：", "结论：", "下一步：")):
        return "请只输出五至六行简短结论，不要输出思考过程"
    if snapshot["ap_id"] not in answer:
        return f"AP 编号应为 {snapshot['ap_id']}"
    forbidden = ("导致", "主因", "已证明", "已证实", "假设不成立", "假设成立", "排除", "非瓶颈", "未达瓶颈", "拥塞阈值", "未形成拥塞")
    for word in forbidden:
        if word in answer:
            return f"删去或改写‘{word}’；单次快照不能证明因果，客户端数也不能证明不存在拥塞"
    if snapshot["rssi_dbm"] > -65 and any(word in answer for word in ("偏弱", "信号弱")):
        return "当前 RSSI 不支持弱信号表述"
    if snapshot["clients"] < 20 and "客户端过多" in answer:
        return "客户端数量本身不能证明过载"
    result = next((x["output"] for x in trace if x["tool"] == "assess_wifi_link" and "error" not in x["output"]), None)
    if not result:
        if any(word in answer for word in ("拥塞", "瓶颈", "原因")):
            return "本次只查询了快照，尚未完成链路分析，请勿诊断瓶颈"
        return None
    busy, backhaul = result["channel_busy_pct"], result["backhaul_util_pct"]
    if busy is None or backhaul is None:
        if "缺" not in answer or not any(word in answer for word in ("不能确定", "无法判断")):
            return "关键占用率缺失，必须说明无法确定是否拥塞"
    elif busy >= 80 and backhaul >= 90:
        if not all(word in answer for word in ("无线", "上联")) or "不能" not in answer:
            return "无线和上联占用都高，必须同时保留两种解释，不能唯一归因"
    elif snapshot["snr_db"] <= 10 and snapshot["rssi_dbm"] <= -80:
        if "优先" not in answer or "信号" not in answer or "当前未见明显瓶颈" in answer:
            return "低 SNR 和弱 RSSI 应优先排查信号质量"
    elif busy >= 80:
        if "优先" not in answer or "无线" not in answer:
            return "高信道忙碌率应优先排查无线占用，不能确认碰撞"
    elif backhaul >= 90:
        if "优先" not in answer or "上联" not in answer:
            return "高上联利用率应优先排查上联，不能确认根因"
    elif "当前未见明显" not in answer and "当前无明显" not in answer:
        return "当前快照没有明显瓶颈，但不能保证其他时段"
    return None


def synthesize(question: str, trace: list[dict]) -> dict:
    if not any("error" not in step["output"] for step in trace):
        return {"answer": grounded_answer(trace), "answer_source": "fallback", "trace": trace, "model": MODEL}
    evidence = json.dumps([{"tool": x["tool"], "output": x["output"]} for x in trace], ensure_ascii=False)
    messages = [{"role": "system", "content": SYNTHESIS},
                {"role": "user", "content": f"问题：{question}\n实际工具结果：{evidence}\n请直接给出六行结论。"}]
    draft = chat(messages, max_tokens=3500).get("content", "").split("</think>")[-1].strip()
    original = draft
    for attempt in range(3):
        issue = review_answer(draft, trace)
        if issue is None:
            return {"answer": draft, "model_draft": original,
                    "answer_source": "model" if attempt == 0 else "model_revised", "trace": trace, "model": MODEL}
        if attempt < 2:
            draft = chat(messages + [{"role": "assistant", "content": draft[:1000]},
                                     {"role": "user", "content": f"上次回答需要修正：{issue}。请依据同一工具结果重写六行；只改结论，不编造数据。"}],
                         max_tokens=3500).get("content", "").split("</think>")[-1].strip()
    return {"answer": grounded_answer(trace), "model_draft": original, "model_revision": draft,
            "answer_source": "fallback", "trace": trace, "model": MODEL}


def run_agent(question: str) -> dict:
    if not question.strip() or len(question) > 500:
        raise ValueError("问题应为 1–500 个字符")
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace = []
    ap_ids = set(re.findall(r"\bAP-[A-Z0-9-]+\b", question.upper()))
    needs_assessment = bool(re.search("为什么|慢|分析|拥塞|正常|查哪里|原因", question))
    for _ in range(5):
        reply = chat(messages, TOOLS)
        calls = reply.get("tool_calls") or []
        if not calls:
            snapshot = next((x["output"] for x in trace if x["tool"] == "get_wifi_metrics" and "error" not in x["output"]), None)
            assessed = any(x["tool"] == "assess_wifi_link" and "error" not in x["output"] for x in trace)
            if not snapshot and len(ap_ids) == 1:
                messages.append({"role": "user", "content": f"你尚未查询数据。现在必须实际调用 get_wifi_metrics，参数 ap_id 为 {next(iter(ap_ids))}；不要直接作答。"})
                continue
            if snapshot and needs_assessment and not assessed:
                messages.append({"role": "assistant", "content": reply.get("content", "")})
                messages.append({"role": "user", "content": "你尚未实际调用 assess_wifi_link。请根据上一工具返回的 SNR、带宽及两个占用率现在调用它；缺失值用 null。"})
                continue
            break
        messages.append(reply)
        for call in calls:
            function = call.get("function", {})
            name, args = function.get("name"), function.get("arguments", {})
            requested = args
            try:
                if name not in FUNCTIONS or not isinstance(args, dict):
                    raise ValueError("未知工具或参数格式错误")
                # Keep tool inputs tied to the user's identifier and measured fields.
                if name == "get_wifi_metrics" and len(ap_ids) == 1:
                    args = {"ap_id": next(iter(ap_ids))}
                if name == "assess_wifi_link":
                    snapshot = next((x["output"] for x in trace if x["tool"] == "get_wifi_metrics" and "error" not in x["output"]), None)
                    if snapshot is None:
                        raise ValueError("须先查询 AP 测量值")
                    args = {key: snapshot[key] for key in ("snr_db", "bandwidth_mhz", "channel_busy_pct", "backhaul_util_pct")}
                result = FUNCTIONS[name](**args)
            except (ValueError, TypeError, KeyError) as exc:
                result = {"error": str(exc)}
            trace.append({"tool": name, "requested_input": requested, "input": args, "output": result})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result, ensure_ascii=False)})
    return synthesize(question, trace)


def run_direct(question: str, snapshot: dict) -> dict:
    # Same factual snapshot as the Agent lookup; no programmatic calculation or tool use.
    messages = [{"role": "system", "content": "你是严谨的校园 Wi-Fi 排障助手。仅依据给定模拟快照，用中文分六行回答：现象、工具、证据、可能原因、结论、下一步。总计不超过 220 汉字。高忙碌率只支持无线拥塞假设，不证明冲突或根因。香农上界只是理想物理层上界，不可与实测速率直接比较证明故障。缺测时必须说无法确定。不要调用工具，不要编造其他数据。"},
                {"role": "user", "content": question + "\n测量快照（模拟）：" + json.dumps(snapshot, ensure_ascii=False)}]
    raw = chat(messages, max_tokens=3000).get("content", "")
    if "</think>" not in raw and len(raw) > 2000:
        raw = chat(messages, max_tokens=6500).get("content", "")
    return {"answer": raw.split("</think>")[-1].strip(), "trace": [], "model": MODEL}
