# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 22:01:45 2025

@author: Eden
"""


import json, re, io, secrets
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st

APP_DIR = Path(__file__).parent.resolve()
BASE_DIR = APP_DIR / "modeling_plus"

try:
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except AttributeError:
    cache_data = st.cache
    cache_resource = st.cache

ZH_LABELS = {
    "IMP_index": "冲动消费指数（目标）",
    "BD_index": "预算纪律指数（目标）",
    "AFF_index": "情绪/社交动因综合指数",
    "BD_F1": "预算纪律因子 F1（执行/遵守）",
    "SAV_F1": "储蓄因子 F1（基础储蓄习惯）",
    "SAV_F2": "储蓄因子 F2（目标与计划）",
    "INV_breadth_z": "投资广度（标准化）",
    "NOISE_randn": "噪声基线（随机）",
    "EF1_score": "执行功能 EF1（计划/注意）",
    "EF2_score": "执行功能 EF2",
    "EF3_score": "执行功能 EF3",
    "EF4_score": "执行功能 EF4（规则/切换）",
    "EF5_score": "执行功能 EF5",
    "EF6_score": "执行功能 EF6",
    "EF7_score": "执行功能 EF7（自控/延迟满足）",
    "EF8_score": "执行功能 EF8（工作记忆）",
    "EF9_score": "执行功能 EF9（情绪自我调节）",
    "EF10_score": "执行功能 EF10",
    "Q7_yes": "是否设定每月预算（Q7）",
    "Q12_yes": "是否记录每日支出（Q12）",
    "Q21_yes": "社交媒体是否诱发过购物欲（Q21）",
    "Q32_yes": "是否有过投资行为（Q32）",
    "Q39_yes": "愿意尝试 AI 投资建议（Q39）",
    "Q43_yes": "偏好“带理由的建议”（Q43）",
    "Q44_yes": "偏好建议型语言（Q44）",
}
EN_LABELS = {
    "IMP_index": "Impulse Spending Index (target)",
    "BD_index": "Budget Discipline Index (target)",
    "AFF_index": "Affect/Social Drivers Index",
    "BD_F1": "Budget Discipline F1 (execution/adherence)",
    "SAV_F1": "Saving Factor F1 (habit)",
    "SAV_F2": "Saving Factor F2 (goals/plans)",
    "INV_breadth_z": "Investment Breadth (z-score)",
    "NOISE_randn": "Noise baseline (random)",
    "EF1_score": "Executive Function EF1 (planning/attention)",
    "EF2_score": "Executive Function EF2",
    "EF3_score": "Executive Function EF3",
    "EF4_score": "Executive Function EF4 (set-shifting/rules)",
    "EF5_score": "Executive Function EF5",
    "EF6_score": "Executive Function EF6",
    "EF7_score": "Executive Function EF7 (self-control/delay)",
    "EF8_score": "Executive Function EF8 (working memory)",
    "EF9_score": "Executive Function EF9 (emotion regulation)",
    "EF10_score": "Executive Function EF10",
    "Q7_yes": "Set monthly budget (Yes/No)",
    "Q12_yes": "Track daily expenses (Yes/No)",
    "Q21_yes": "Social media triggered desire (Yes/No)",
    "Q32_yes": "Ever invested before (Yes/No)",
    "Q39_yes": "Willing to try AI investment advice (Yes/No)",
    "Q43_yes": "Prefer advice with reasons (Yes/No)",
    "Q44_yes": "Prefer advisory tone (Yes/No)",
}

THRESHOLDS_FALLBACK = {
    "IMP_index": (-0.10, 0.25),
    "BD_index": (-0.10, 0.25),
}

def zh(name: str) -> str:
    if name.startswith("INT__"):
        parts = name.split("__")
        if len(parts) >= 4 and parts[2] == "x":
            a, b = parts[1], parts[3]
            return f"交互：{ZH_LABELS.get(a, a)} × {ZH_LABELS.get(b, b)}"
    return ZH_LABELS.get(name, name)

def en(name: str) -> str:
    if name.startswith("INT__"):
        parts = name.split("__")
        if len(parts) >= 4 and parts[2] == "x":
            a, b = parts[1], parts[3]
            return f"Interaction: {EN_LABELS.get(a, a)} × {EN_LABELS.get(b, b)}"
    return EN_LABELS.get(name, name)

def bi(name: str) -> str:
    return f"{zh(name)} / {en(name)}（{name}）"

@cache_data(show_spinner=False)
def load_json(p: Path):
    if not p.exists():
        return None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except Exception:
            pass
    return None

def find_model_path(exp_dir: Path):
    for name in ["model.pkl","model.joblib","pipeline.pkl","clf.pkl","estimator.pkl"]:
        p = exp_dir / name
        if p.exists():
            return p
    for p in exp_dir.glob("*.pkl"):
        return p
    for p in exp_dir.glob("*.joblib"):
        return p
    return None

@cache_resource(show_spinner=False)
def load_model_any(p: Path):
    try:
        m = joblib.load(p)
        return m, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def safe_ranges(schema_cols, ranges):
    if not isinstance(ranges, dict):
        ranges = {}
    def guess(name):
        if name.endswith("_yes"):
            return {"lo": 0, "hi": 1, "default": 0, "step": 1}
        if name.endswith("_z"):
            return {"lo": -2.0, "hi": 2.0, "default": 0.0, "step": 0.1}
        return {"lo": -1.5, "hi": 1.5, "default": 0.0, "step": 0.1}
    out = {}
    for f in schema_cols:
        out[f] = ranges.get(f, guess(f))
    return out

def read_columns_from_readme(exp_dir: Path):
    p = exp_dir / "README_export.txt"
    if not p.exists():
        return []
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    m = re.search(r"(columns?|cols?)\s*[:=]\s*(\[[^\]]+\])", txt, flags=re.I)
    if not m:
        m = re.search(r"\[[^\]]+\]", txt)
        if not m:
            return []
    block = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(0)
    try:
        arr = json.loads(block.replace("'", '"'))
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr) and len(arr) > 0:
            return arr
    except Exception:
        pass
    items = [x.strip().strip("'\"") for x in block.strip("[]").split(",")]
    return [x for x in items if x]

def infer_columns(exp_dir: Path, schema: dict, model):
    cols = []
    ranges0 = {}
    if isinstance(schema, dict):
        for k in ["feature_order","columns","cols","features","feature_names","featureNames","input_columns","input_cols","fields","variables","predictors","X_columns","x_columns","raw_columns"]:
            v = schema.get(k)
            if isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) > 0:
                cols = v
                ranges0 = schema.get("ranges") if isinstance(schema.get("ranges"), dict) else {}
                break
        if not cols and isinstance(schema.get("ranges"), dict) and len(schema["ranges"]) > 0:
            cols = list(schema["ranges"].keys()); ranges0 = schema["ranges"]
    if not cols:
        cols = read_columns_from_readme(exp_dir)
    if not cols and model is not None:
        names = getattr(model, "feature_names_in_", None)
        if names is None and hasattr(model, "named_steps"):
            for step in list(model.named_steps.values())[::-1]:
                names = getattr(step, "feature_names_in_", None)
                if names is not None:
                    break
        if names is not None:
            cols = [str(x) for x in list(names)]
    return cols, safe_ranges(cols, ranges0)

def load_bundle(target: str):
    exp_dir = BASE_DIR / target / "export"
    schema = load_json(exp_dir / "schema.json") or {}
    inter_map = load_json(exp_dir / "interactions.json") or {}
    errors = []
    if not exp_dir.exists():
        errors.append(f"{target}：导出目录不存在：{exp_dir}")
    model_path = find_model_path(exp_dir)
    model = None
    if model_path is None:
        errors.append(f"{target}：未在 {exp_dir} 找到模型文件")
    else:
        model, err = load_model_any(model_path)
        if model is None:
            errors.append(f"{target}：无法加载模型 {model_path.name} —— {err}")
    cols, ranges = infer_columns(exp_dir, schema, model)
    if not cols:
        errors.append(f"{target}：无法从 schema、README 或模型中推断特征列表")
    simple_feats = schema.get("simple_feats")
    if not isinstance(simple_feats, list) or not simple_feats:
        simple_feats = cols[:8] if len(cols) >= 8 else cols
    thresholds = None
    if isinstance(schema.get("thresholds"), dict):
        low = schema["thresholds"].get("low")
        high = schema["thresholds"].get("high")
        if isinstance(low, (int,float)) and isinstance(high, (int,float)):
            thresholds = (float(low), float(high))
    return {
        "model": model,
        "schema_cols": cols,
        "simple_feats": simple_feats,
        "ranges": ranges,
        "inter_map": inter_map,
        "thresholds": thresholds or THRESHOLDS_FALLBACK.get(target, (-0.10, 0.25)),
        "errors": errors,
        "export_dir": str(exp_dir),
    }

def apply_interactions(X: pd.DataFrame, inter_map: dict):
    if not inter_map:
        return X
    X2 = X.copy()
    for name, pair in inter_map.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        a, b = pair
        if a in X2.columns and b in X2.columns:
            X2[name] = pd.to_numeric(X2[a], errors="coerce") * pd.to_numeric(X2[b], errors="coerce")
    return X2

def make_input_row(schema_cols, user_vals: dict, inter_map: dict):
    row = {}
    for c in schema_cols:
        v = user_vals.get(c, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        row[c] = v
    X = pd.DataFrame([row])
    return apply_interactions(X, inter_map)

def label_and_actions(task: str, y_pred: float, thresholds):
    low_thr, high_thr = thresholds
    acts = []
    if task == "IMP_index":
        if y_pred <= low_thr:
            label = "冲动偏低（自控较好） / Low impulse"
            acts = ["维持记录日常支出 / Keep expense logging", "保留发薪或大额提醒 / Keep payday & large-purchase alerts", "适度自我奖励 / Occasional self-reward"]
        elif y_pred <= high_thr:
            label = "中等（可控区间） / Moderate"
            acts = ["购物车等待 24 小时再决定 / 24h wait rule", "对高频诱因设上限 / Caps for frequent triggers", "每周复盘 5 笔最大或后悔支出 / Weekly review top-5 spends"]
        else:
            label = "冲动偏高（需要干预） / High impulse"
            acts = ["先加购物车 24 小时后再决定 / Add-to-cart cooling-off", "情绪性消费用低成本替代 / Low-cost emotional substitutes", "锁定 1 个高频破功场景设置提醒与限额 / Target one trigger with limits"]
    else:
        if y_pred <= low_thr:
            label = "预算纪律较弱 / Weak discipline"
            acts = ["只跟三类账本：餐饮/通勤/娱乐 / Track 3 core categories", "每周 10 分钟对照预算上限 / Weekly 10-min cap check", "储蓄改为发薪日自动扣款 / Auto-save on payday"]
        elif y_pred <= high_thr:
            label = "预算纪律一般 / Moderate discipline"
            acts = ["为各类支出设软上限 / Soft caps & alerts", "坚持一周复盘并给予小奖励 / Weekly review with small reward"]
        else:
            label = "预算纪律良好 / Strong discipline"
            acts = ["维持现状 / Maintain", "总结 3 条个人黄金法则 / Write 3 golden rules"]
    return label, acts

def ensure_dir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def append_row_to_csv(path: Path, row: dict):
    try:
        if path.exists():
            try:
                df = pd.read_csv(path, encoding="utf-8")
            except Exception:
                df = pd.read_csv(path, encoding="utf-8-sig")
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(path, index=False, encoding="utf-8")
        return True, str(path)
    except Exception as e:
        return False, str(e)

st.set_page_config(page_title="Personal Finance Assistant (Demo)", layout="centered")
st.title("AI 个人理财助手 · AI Personal Finance Assistant")

with st.expander("使用说明 / How to use", expanded=True):
    st.markdown("""
本页用于学术演示，根据你的输入预测两个指数并返回轻量建议。  
This page is for academic demo. It predicts two indices and returns lightweight advice.

• IMP_index（冲动消费指数） / higher → more impulsive  
• BD_index（预算纪律指数） / higher → stronger discipline

结果与体验反馈会汇总到同一个 CSV（responses_log.csv）。作者可在页面底部下载。
""")

if "log_rows" not in st.session_state:
    st.session_state["log_rows"] = []
if "result_links" not in st.session_state:
    st.session_state["result_links"] = []

task = st.sidebar.selectbox("选择要测试的目标 / Choose target", ["IMP_index", "BD_index"], index=0)
bundle = load_bundle(task)

if bundle["errors"]:
    st.error("加载资源时遇到问题 / Errors during loading:\n- " + "\n- ".join(bundle["errors"]))
    st.stop()

model = bundle["model"]
schema_cols = bundle["schema_cols"]
simple_feats = bundle["simple_feats"]
ranges = bundle["ranges"]
inter_map = bundle["inter_map"]
export_dir = bundle["export_dir"]
thresholds = bundle["thresholds"]

st.header("简易模式 / Simple Mode")
st.caption("填写少量关键问题；0≈平均。Provide a few key inputs; 0 ≈ average.")

user_simple = {}
cols2 = st.columns(2)
for i, f in enumerate(simple_feats):
    r = ranges.get(f, {"lo": -1.5, "hi": 1.5, "default": 0.0, "step": 0.1})
    title = bi(f)
    if f.endswith("_yes"):
        user_simple[f] = cols2[i % 2].selectbox(title, [0, 1], index=int(r.get("default", 0) > 0), format_func=lambda x: "是/Yes" if x == 1 else "否/No", key=f"simp_{f}")
    else:
        user_simple[f] = cols2[i % 2].slider(title, float(r["lo"]), float(r["hi"]), float(r["default"]), step=float(r["step"]), key=f"simp_{f}")

if st.button("用简易模式预测 / Predict (Simple)", type="primary"):
    X = make_input_row(schema_cols, user_simple, inter_map)
    try:
        y_pred = float(model.predict(X)[0])
    except Exception as e:
        st.error(f"模型预测失败 / Prediction failed: {e}")
        st.stop()
    label, actions = label_and_actions(task, y_pred, thresholds)
    rid = secrets.token_hex(4)
    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "result",
        "mode": "simple",
        "task": task,
        "row_id": rid,
        "y_pred": y_pred,
        "label": label,
        "inputs_json": json.dumps(user_simple, ensure_ascii=False),
    }
    st.session_state["log_rows"].append(row)
    st.session_state["result_links"].append((f"Simple @ {row['ts']} [{task}]", rid))
    st.success("简易模式：预测完成 / Done")
    st.metric("预测数值 / Predicted", f"{y_pred:.3f}")
    st.metric("定性判断 / Interpretation", label)
    with st.expander("建议清单 / Action list", expanded=True):
        for i, a in enumerate(actions, 1):
            st.write(f"{i}. {a}")
    st.info(f"已记录到会话日志（row_id={rid}）")

st.divider()

st.header("专家模式 / Expert Mode")
st.caption("逐项可控；0 表示平均水平。Fine-grained control; 0 ≈ average.")

user_adv = {}
for f in sorted(schema_cols):
    r = ranges.get(f, {"lo": -1.5, "hi": 1.5, "default": 0.0, "step": 0.1})
    title = bi(f)
    if f.endswith("_yes"):
        user_adv[f] = st.selectbox(title, [0, 1], index=int(r.get("default", 0) > 0), format_func=lambda x: "是/Yes" if x == 1 else "否/No", key=f"adv_{f}")
    elif f.endswith("_z"):
        user_adv[f] = st.slider(title, float(r["lo"]), float(r["hi"]), float(r["default"]), step=float(r["step"]), key=f"adv_{f}")
    else:
        user_adv[f] = st.number_input(title, value=float(r.get("default", 0.0)), step=0.1, key=f"adv_{f}")

if st.button("用专家模式预测 / Predict (Expert)"):
    X = make_input_row(schema_cols, user_adv, inter_map)
    try:
        y_pred = float(model.predict(X)[0])
    except Exception as e:
        st.error(f"模型预测失败 / Prediction failed: {e}")
        st.stop()
    label, actions = label_and_actions(task, y_pred, thresholds)
    rid = secrets.token_hex(4)
    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "result",
        "mode": "expert",
        "task": task,
        "row_id": rid,
        "y_pred": y_pred,
        "label": label,
        "inputs_json": json.dumps(user_adv, ensure_ascii=False),
    }
    st.session_state["log_rows"].append(row)
    st.session_state["result_links"].append((f"Expert @ {row['ts']} [{task}]", rid))
    st.success("专家模式：预测完成 / Done")
    st.metric("预测数值 / Predicted", f"{y_pred:.3f}")
    st.metric("定性判断 / Interpretation", label)
    with st.expander("建议清单 / Action list", expanded=True):
        for i, a in enumerate(actions, 1):
            st.write(f"{i}. {a}")
    st.info(f"已记录到会话日志（row_id={rid}）")

st.divider()
st.subheader("你的体验反馈（可选） / Optional UX feedback")

options = [f"{t}（{rid}）" for (t, rid) in st.session_state.get("result_links", [])]
link_choice = st.selectbox("关联结果 / Link to result", options, index=len(options)-1 if options else 0) if options else st.selectbox("关联结果 / Link to result", ["无 / None"])
link_rid = None
if options:
    link_rid = st.session_state["result_links"][options.index(link_choice)][1]

c1, c2, c3 = st.columns(3)
easy = c1.slider("易用 / Ease (1–5)", 1, 5, 4)
clear = c2.slider("易懂 / Clarity (1–5)", 1, 5, 4)
trust = c3.slider("可信 / Trust (1–5)", 1, 5, 4)
comment = st.text_input("其他建议或吐槽 / Other comments")

if st.button("提交反馈 / Submit feedback"):
    fb = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "kind": "feedback",
        "task": task,
        "link_row_id": link_rid or "",
        "ease": int(easy),
        "clarity": int(clear),
        "trust": int(trust),
        "comment": str(comment or ""),
    }
    st.session_state["log_rows"].append(fb)
    st.success("反馈已加入会话汇总 / Feedback added to session log")

st.divider()
with st.expander("其它 / Utilities", expanded=False):
    df = pd.DataFrame(st.session_state["log_rows"])
    st.write("会话汇总（结果+反馈）/ Session combined log")
    st.dataframe(df, use_container_width=True)
    csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
    st.download_button("下载本次会话 CSV（作者） / Download session CSV", data=csv_bytes, file_name="responses_log.csv", mime="text/csv")
    if st.button("尝试写入到工程目录（云端可忽略） / Try write to project file"):
        out_path = APP_DIR / "responses_log.csv"
        ok, msg = append_row_to_csv(out_path, {}) if df.empty else append_row_to_csv(out_path, df.iloc[-1].to_dict())
        if ok and not df.empty:
            try:
                if out_path.exists():
                    base = pd.read_csv(out_path, encoding="utf-8")
                    base = pd.concat([base, df.iloc[:-1]], ignore_index=True)
                    base.to_csv(out_path, index=False, encoding="utf-8")
            except Exception:
                pass
        st.info(f"写入结果：{msg}")
