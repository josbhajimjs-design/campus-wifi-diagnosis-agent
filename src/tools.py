"""Two bounded, read-only tools for the simulated campus Wi-Fi snapshots."""
import json
import math
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data" / "snapshots.json"


def get_wifi_metrics(ap_id: str) -> dict:
    """Look up measured fields for one AP; never return a diagnosis."""
    if not isinstance(ap_id, str) or not ap_id.startswith("AP-") or len(ap_id) > 32:
        raise ValueError("AP 编号必须是 AP- 开头且不超过 32 个字符")
    records = json.loads(DATA.read_text(encoding="utf-8"))
    if ap_id not in records:
        raise ValueError(f"未找到 AP：{ap_id}")
    return {"ap_id": ap_id, **records[ap_id]}


def assess_wifi_link(snr_db: float, bandwidth_mhz: float, channel_busy_pct: float | None,
                     backhaul_util_pct: float | None) -> dict:
    """Compute an ideal Shannon bound and expose measured resource occupancy."""
    if not all(type(x) in (int, float) and math.isfinite(x) for x in (snr_db, bandwidth_mhz)):
        raise ValueError("SNR 和带宽必须是有限数字")
    if not -20 <= snr_db <= 60 or not 1 <= bandwidth_mhz <= 320:
        raise ValueError("SNR 或带宽超出合理范围")
    for name, value in (("信道忙碌率", channel_busy_pct), ("上联利用率", backhaul_util_pct)):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError(f"{name}应为 0–100 的有限数字或 null")
    capacity_mbps = bandwidth_mhz * math.log2(1 + 10 ** (snr_db / 10))
    return {
        "shannon_upper_bound_mbps": round(capacity_mbps, 2),
        "channel_busy_pct": channel_busy_pct,
        "backhaul_util_pct": backhaul_util_pct,
        "note": "香农值是理想物理层上界，不等于 Wi-Fi 实测吞吐量；忙碌率和上联利用率仅提示排查方向。",
    }
