"""Small runnable check for the nontrivial numerical and input-boundary logic."""
import math
from src.tools import assess_wifi_link, get_wifi_metrics
from src.agent import grounded_answer, review_answer


assert get_wifi_metrics("AP-DORM-02")["snr_db"] == 30
assert math.isclose(assess_wifi_link(30, 40, 89, 38)["shannon_upper_bound_mbps"], 398.69, abs_tol=0.01)
assert assess_wifi_link(29, 40, None, None)["channel_busy_pct"] is None
missing = [{"tool": "get_wifi_metrics", "output": get_wifi_metrics("AP-CLASS-04")},
           {"tool": "assess_wifi_link", "output": assess_wifi_link(29, 40, None, None)}]
assert "不能确定" in grounded_answer(missing)
assert "缺失%" not in grounded_answer(missing)
conflicting = [{"tool": "get_wifi_metrics", "output": get_wifi_metrics("AP-ENG-06")},
               {"tool": "assess_wifi_link", "output": assess_wifi_link(31, 40, 87, 94)}]
assert "两种瓶颈" in grounded_answer(conflicting)
bad_conclusion = "现象：AP-ENG-06 慢。\n证据：87% 忙碌。\n可能原因：无线拥塞。\n结论：无线是主因。\n下一步：换信道。"
assert review_answer(bad_conclusion, conflicting) is not None
try:
    assess_wifi_link(30, 40, 101, 38)
except ValueError:
    pass
else:
    raise AssertionError("Invalid occupancy was accepted")
try:
    get_wifi_metrics("AP-NOT-FOUND")
except ValueError:
    pass
else:
    raise AssertionError("Unknown AP was accepted")
print("工具检查通过")
