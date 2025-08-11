# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 21:30:48 2025

@author: Eden
"""

# app.py
import json, re, uuid
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Finance Assistant", layout="centered")

def _sc():
    return st.session_state

def _now():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def _init_state():
    if "session_log" not in _sc():
        _sc().session_log = []   # 每条都是 dict，event ∈ {"predict","feedback"}

_init_state()

BASE_DIR = Path(__file__).parent / "modeling_plus"

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
    "EF4_score": "执行功能 EF4（规则/切换）",
    "EF7_score": "执行功能 EF7（自控/延迟满足）",
    "EF9_score": "执行功能 EF9（情绪自我调节）",
    "Q7_yes": "是否设定每月预算（Q7）",
    "Q32_yes": "是否有过投资行为（Q32）",
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
    "EF1_score": "EF1 (planning/attention)",
    "EF4_score": "EF4 (set-shifting/rules)",
    "EF7_score": "EF7 (self-control/delay)",
    "EF9_score": "EF9 (emotion regulation)",
    "Q7_yes": "Set monthly budget (Yes/No)",
    "Q32_yes": "Ever invested before (Yes/No)",
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

def read_json(p: Path):
    if not p.exists(): return None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(p.read_text(encoding=enc))
        except Exception:
            pass
    return None

def read_columns_from_readme(exp_dir: Path):
    p = exp_dir / "README_export.txt"
    if not p.exists(): return []
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    m = re.search(r"(columns?|cols?)\s*[:=]\s*(\[[^\]]+\])", txt, flags=re.I)
    if not m:
        m = re.search(r"\[[^\]]+\]", txt)
        if not m: return []
    block = m.group(2) if (m.lastindex and m.lastindex >= 2) else m.group(0)
    try:
        arr = json.loads(block.replace("'", '"'))
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except Exception:
        pass
    items = [x.strip().strip("'\"") for x in block.strip("[]").split(",")]
    return [x for x in items if x]

def find_model_path(exp_dir: Path):
    for name in ["model.pkl","model.joblib","pipeline.pkl","clf.pkl","estimator.pkl"]:
        p = exp_dir / name
        if p.exists(): return p
    for p in exp_dir.glob("*.pkl"):
        return p
    for p in exp_dir.glob("*.joblib"):
        return p
    return None

@st.cache_resource(show_spinner=False)
def load_model(p: Path):
    try:
        return joblib.load(p), None
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

def infer_columns(exp_dir: Path, schema: dict, model):
    cols, ranges0 = [], {}
    if isinstance(schema, dict):
        for k in ["feature_order","columns","cols","features","feature_names","featureNames","input_columns","input_cols","fields","variables","predictors","X_columns","x_columns","raw_columns"]:
            v = schema.get(k)
            if isinstance(v, list) and all(isinstance(x, str) for x in v) and len(v) > 0:
                cols = v
                ranges0 = schema.get("ranges") if isinstance(schema.get("ranges"), dict) else {}
                break
        if not cols and isinstance(schema.get("ranges"), dict) and schema["ranges"]:
            cols = list(schema["ranges"].keys()); ranges0 = schema["ranges"]
    if not cols:
        cols = read_columns_from_readme(exp_dir)
    if not cols and model is not None:
        names = getattr(model, "feature_names_in_", None)
        if names is None and hasattr(model, "named_steps"):
            for step in list(model.named_steps.values())[::-1]:
                names = getattr(step, "feature_names_in_", None)
                if names is not None: break
        if names is not None:
            cols = [str(x) for x in list(names)]
    return cols, safe_ranges(cols, ranges0)

def load_bundle(target: str):
    exp_dir = BASE_DIR / target / "export"
    schema = read_json(exp_dir / "schema.json") or {}
    inter_map = read_json(exp_dir / "interactions.json") or {}
    errs = []
    if not exp_dir.exists():
        errs.append(f"{target}：未找到导出目录 {exp_dir}")

    model_path = find_model_path(exp_dir)
    model = None
    if model_path is None:
        errs.append(f"{target}：导出目录中未找到模型文件")
    else:
        model, err = load_model(model_path)
        if model is None:
            errs.append(f"{target}：无法加载模型 —— {err}")

    cols, ranges = infer_columns(exp_dir, schema, model)
    if not cols:
        errs.append(f"{target}：无法从 schema/README/模型推断特征列表")

    simple_feats = schema.get("simple_feats")
    if not isinstance(simple_feats, list) or not simple_feats:
        simple_feats = cols[:8] if len(cols) >= 8 else cols

    return {
        "model": model,
        "schema_cols": cols,
        "simple_feats": simple_feats,
        "ranges": ranges,
        "inter_map": inter_map,
        "errors": errs,
    }

def apply_interactions(X: pd.DataFrame, inter_map: dict):
    if not inter_map: return X
    X2 = X.copy()
    for name, pair in inter_map.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2: continue
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

def label_and_actions(task: str, y: float):
    if task == "IMP_index":
        if y <= -0.5:
            label = "冲动偏低（自控较好） / Low impulse"
            acts = ["维持记录日常支出 / Keep expense logging",
                    "保留发薪或大额提醒 / Keep payday & large-purchase alerts",
                    "适度自我奖励 / Occasional self-reward"]
        elif y <= 0.5:
            label = "中等（可控区间） / Moderate"
            acts = ["购物车等待 24 小时再决定 / 24h wait rule",
                    "对高频诱因设上限 / Caps for frequent triggers",
                    "每周复盘 5 笔最大或后悔支出 / Weekly review top-5 spends"]
        else:
            label = "冲动偏高（需要干预） / High impulse"
            acts = ["先加购物车 24 小时后再决定 / Add-to-cart cooling-off",
                    "情绪性消费用低成本替代 / Low-cost emotional substitutes",
                    "锁定 1 个高频破功场景设置提醒与限额 / Target one trigger with limits"]
    else:  # BD_index
        if y <= -0.3:
            label = "预算纪律较弱 / Weak discipline"
            acts = ["只跟三类账本：餐饮/通勤/娱乐 / Track 3 core categories",
                    "每周 10 分钟对照预算上限 / Weekly 10-min cap check",
                    "储蓄改为发薪日自动扣款 / Auto-save on payday"]
        elif y <= 0.4:
            label = "预算纪律一般 / Moderate discipline"
            acts = ["为各类支出设软上限 / Soft caps & alerts",
                    "坚持一周复盘并给予小奖励 / Weekly review with small reward"]
        else:
            label = "预算纪律良好 / Strong discipline"
            acts = ["维持现状 / Maintain",
                    "总结 3 条个人黄金法则 / Write 3 golden rules"]
    return label, acts

def add_predict_log(task, mode, y_pred, label, inputs: dict):
    rid = str(uuid.uuid4())[:8]
    _sc().session_log.append({
        "event": "predict",
        "row_id": rid,
        "ts": _now(),
        "task": task,
        "mode": mode,
        "y_pred": float(y_pred),
        "label": label,
        "inputs_json": json.dumps(inputs, ensure_ascii=False),
    })
    st.info(f"已记录到会话日志（row_id={rid}）")
    return rid

def add_feedback_log(link_row_id, easy, clarity, trust, comment):
    _sc().session_log.append({
        "event": "feedback",
        "row_id": link_row_id,
        "ts": _now(),
        "easy": int(easy),
        "clarity": int(clarity),
        "trust": int(trust),
        "comment": str(comment or ""),
    })
    st.success("反馈已加入会话日志")

def render_predict_block(task, bundle, mode: str):
    model = bundle["model"]
    schema_cols = bundle["schema_cols"]
    ranges = bundle["ranges"]
    inter_map = bundle["inter_map"]

    if mode == "simple":
        st.header("简易模式 / Simple Mode")
        st.caption("填写少量关键问题；不确定可保持默认值（0≈平均）。")
        simple_feats = bundle["simple_feats"]
        user_vals = {}
        cols2 = st.columns(2)
        for i, f in enumerate(simple_feats):
            r = ranges.get(f, {"lo": -1.5, "hi": 1.5, "default": 0.0, "step": 0.1})
            title = bi(f)
            if f.endswith("_yes"):
                user_vals[f] = cols2[i % 2].selectbox(
                    title, [0, 1], index=int(r.get("default", 0) > 0),
                    format_func=lambda x: "是/Yes" if x == 1 else "否/No",
                    key=f"simp_{task}_{f}"
                )
            else:
                user_vals[f] = cols2[i % 2].slider(
                    title, float(r["lo"]), float(r["hi"]), float(r["default"]),
                    step=float(r["step"]), key=f"simp_{task}_{f}"
                )

        if st.button("用简易模式预测 / Predict (Simple)", type="primary", key=f"btn_simp_{task}"):
            X = make_input_row(schema_cols, user_vals, inter_map)
            try:
                y_pred = float(model.predict(X)[0])
            except Exception as e:
                st.error(f"预测失败：{e}")
                return
            label, actions = label_and_actions(task, y_pred)
            st.success("简易模式：预测完成 / Done")
            st.metric("预测数值 / Predicted", f"{y_pred:.3f}")
            st.metric("定性判断 / Interpretation", label)
            with st.expander("建议清单 / Action list（Simple）", expanded=True):
                for i, a in enumerate(actions, 1): st.write(f"{i}. {a}")
            add_predict_log(task, "simple", y_pred, label, user_vals)

    else:
        st.header("专家模式 / Expert Mode")
        st.caption("逐项可控；0 表示平均水平。")
        user_vals = {}
        for f in sorted(schema_cols):
            r = ranges.get(f, {"lo": -1.5, "hi": 1.5, "default": 0.0, "step": 0.1})
            title = bi(f)
            if f.endswith("_yes"):
                user_vals[f] = st.selectbox(
                    title, [0, 1], index=int(r.get("default", 0) > 0),
                    format_func=lambda x: "是/Yes" if x == 1 else "否/No",
                    key=f"adv_{task}_{f}"
                )
            elif f.endswith("_z"):
                user_vals[f] = st.slider(
                    title, float(r["lo"]), float(r["hi"]), float(r["default"]),
                    step=float(r["step"]), key=f"adv_{task}_{f}"
                )
            else:
                user_vals[f] = st.number_input(
                    title, value=float(r.get("default", 0.0)), step=0.1, key=f"adv_{task}_{f}"
                )

        if st.button("用专家模式预测 / Predict (Expert)", type="primary", key=f"btn_adv_{task}"):
            X = make_input_row(schema_cols, user_vals, inter_map)
            try:
                y_pred = float(model.predict(X)[0])
            except Exception as e:
                st.error(f"预测失败：{e}")
                return
            label, actions = label_and_actions(task, y_pred)
            st.success("专家模式：预测完成 / Done")
            st.metric("预测数值 / Predicted", f"{y_pred:.3f}")
            st.metric("定性判断 / Interpretation", label)
            with st.expander("建议清单 / Action list（Expert）", expanded=True):
                for i, a in enumerate(actions, 1): st.write(f"{i}. {a}")
            add_predict_log(task, "expert", y_pred, label, user_vals)

def render_feedback_block():
    st.subheader("你的体验反馈（可选） / Optional UX feedback")
    preds = [r for r in _sc().session_log if r.get("event") == "predict"]
    if not preds:
        st.info("请先完成一次预测，再提交反馈。")
        return
    options = []
    for r in preds[::-1]:
        tag = f"{r['mode'].capitalize()} @ {r['ts']} [{r['task']}]"
        options.append((tag, r["row_id"]))
    tag2rid = {tag: rid for tag, rid in options}
    pick = st.selectbox("关联结果 / Link to result", list(tag2rid.keys()), index=0, key="fb_link_pick")
    link_row_id = tag2rid[pick]

    c1, c2, c3 = st.columns(3)
    easy  = c1.slider("易用 / Ease (1–5)", 1, 5, 4, key="fb_ease")
    clear = c2.slider("易懂 / Clarity (1–5)", 1, 5, 4, key="fb_clr")
    trust = c3.slider("可信 / Trust (1–5)", 1, 5, 4, key="fb_tru")
    comment = st.text_input("其他建议或吐槽 / Other comments", key="fb_cmt")

    if st.button("提交反馈 / Submit feedback", key="btn_fb"):
        add_feedback_log(link_row_id, easy, clear, trust, comment)

def df_session_log():
    if not _sc().session_log:
        return pd.DataFrame()
    return pd.DataFrame(_sc().session_log)

def render_utilities():
    with st.expander("其它 / Utilities", expanded=False):
        df = df_session_log()
        if df.empty:
            st.info("本次会话暂无可下载数据。")
        else:
            csv_all = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "下载本次会话全部数据 CSV / Download session CSV",
                csv_all, file_name=f"session_log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", key="dl_all"
            )
            df_fb = df[df["event"]=="feedback"]
            if not df_fb.empty:
                csv_fb = df_fb.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "仅下载反馈 CSV / Download feedback CSV",
                    csv_fb, file_name=f"feedback_only_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv", key="dl_fb"
                )
        if st.button("清空当前会话数据 / Clear session", key="clear_sess"):
            _sc().session_log = []
            st.success("已清空。")

st.title("Finance Assistant")

with st.expander("使用说明 / How to use", expanded=True):
    st.markdown("""
本页用于学术演示，将根据你的输入预测两个指数并返回轻量建议。  
This page is for academic demo. It predicts two indices and returns lightweight advice.

• IMP_index（冲动消费指数）/ higher → more impulsive  
• BD_index（预算纪律指数）/ higher → stronger discipline

不会写入服务器磁盘。所有预测与反馈暂存在**本次会话内存**，可在页面底部“其它 / Utilities”**下载 CSV**。
""")

task = st.sidebar.selectbox("选择要测试的目标 / Choose target", ["IMP_index", "BD_index"], index=0)
bundle = load_bundle(task)

if bundle["errors"]:
    st.error("加载资源时遇到问题 / Errors during loading:\n- " + "\n- ".join(bundle["errors"]))
else:
    render_predict_block(task, bundle, mode="simple")
    st.divider()
    render_predict_block(task, bundle, mode="expert")
    st.divider()
    render_feedback_block()
    st.divider()
    render_utilities()
