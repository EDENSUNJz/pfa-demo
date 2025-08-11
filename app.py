# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 19:23:28 2025

@author: Eden
"""


import json, re, os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st


BASE_DIR = Path("modeling_plus")  # 云端/本地统一用相对路径


try:
    cache_data = st.cache_data
    cache_resource = st.cache_resource
except AttributeError:
    cache_data = st.cache
    cache_resource = st.cache


ZH = {
    "IMP_index":"冲动消费指数（目标）","BD_index":"预算纪律指数（目标）",
    "AFF_index":"情绪/社交动因综合指数","BD_F1":"预算纪律因子 F1（执行/遵守）",
    "SAV_F1":"储蓄因子 F1（基础储蓄习惯）","SAV_F2":"储蓄因子 F2（目标与计划）",
    "INV_breadth_z":"投资广度（标准化）","NOISE_randn":"噪声基线（随机）",
    "EF1_score":"执行功能 EF1（计划/注意）","EF2_score":"执行功能 EF2",
    "EF3_score":"执行功能 EF3","EF4_score":"执行功能 EF4（规则/切换）",
    "EF5_score":"执行功能 EF5","EF6_score":"执行功能 EF6",
    "EF7_score":"执行功能 EF7（自控/延迟满足）",
    "EF8_score":"执行功能 EF8（工作记忆）",
    "EF9_score":"执行功能 EF9（情绪自我调节）",
    "EF10_score":"执行功能 EF10",
    "Q7_yes":"是否设定每月预算（Q7）","Q12_yes":"是否记录每日支出（Q12）",
    "Q21_yes":"社交媒体是否诱发过购物欲（Q21）","Q32_yes":"是否有过投资行为（Q32）",
    "Q39_yes":"愿意尝试 AI 投资建议（Q39）","Q43_yes":"偏好“带理由的建议”（Q43）",
    "Q44_yes":"偏好建议型语言（Q44）",
}
EN = {
    "IMP_index":"Impulse Spending Index (target)","BD_index":"Budget Discipline Index (target)",
    "AFF_index":"Affect/Social Drivers Index","BD_F1":"Budget Discipline F1 (execution/adherence)",
    "SAV_F1":"Saving Factor F1 (habit)","SAV_F2":"Saving Factor F2 (goals/plans)",
    "INV_breadth_z":"Investment Breadth (z-score)","NOISE_randn":"Noise baseline (random)",
    "EF1_score":"Executive Function EF1 (planning/attention)","EF2_score":"Executive Function EF2",
    "EF3_score":"Executive Function EF3","EF4_score":"Executive Function EF4 (set-shifting/rules)",
    "EF5_score":"Executive Function EF5","EF6_score":"Executive Function EF6",
    "EF7_score":"Executive Function EF7 (self-control/delay)",
    "EF8_score":"Executive Function EF8 (working memory)",
    "EF9_score":"Executive Function EF9 (emotion regulation)",
    "EF10_score":"Executive Function EF10",
    "Q7_yes":"Set monthly budget (Yes/No)","Q12_yes":"Track daily expenses (Yes/No)",
    "Q21_yes":"Social media triggered desire (Yes/No)","Q32_yes":"Ever invested before (Yes/No)",
    "Q39_yes":"Willing to try AI investment advice (Yes/No)","Q43_yes":"Prefer advice with reasons (Yes/No)",
    "Q44_yes":"Prefer advisory tone (Yes/No)",
}
def bi(name:str)->str:
    if name.startswith("INT__"):
        a,b = name.split("__")[1], name.split("__")[3]
        return f"交互：{ZH.get(a,a)} × {ZH.get(b,b)} / Interaction: {EN.get(a,a)} × {EN.get(b,b)}（{name}）"
    return f"{ZH.get(name,name)} / {EN.get(name,name)}（{name}）"


@cache_data(show_spinner=False)
def load_json(p: Path):
    if p.exists():
        for enc in ("utf-8","utf-8-sig"):
            try:
                return json.loads(p.read_text(encoding=enc))
            except Exception:
                pass
    return None

def find_model_path(exp_dir: Path):
    for n in ["model.pkl","model.joblib","pipeline.pkl","clf.pkl","estimator.pkl"]:
        p = exp_dir/n
        if p.exists(): return p
    for p in list(exp_dir.glob("*.pkl"))+list(exp_dir.glob("*.joblib")):
        return p
    return None

@cache_resource(show_spinner=False)
def load_model_any(p: Path):
    try:
        return joblib.load(p), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def safe_ranges(schema_cols, ranges):
    if not isinstance(ranges, dict): ranges={}
    def guess(n):
        if n.endswith("_yes"): return dict(lo=0,hi=1,default=0,step=1)
        if n.endswith("_z"):  return dict(lo=-2,hi=2,default=0,step=0.1)
        return dict(lo=-1,hi=1.5,default=0,step=0.1)
    return {c: ranges.get(c, guess(c)) for c in schema_cols}

def read_cols_from_readme(exp_dir: Path):
    p = exp_dir/"README_export.txt"
    if not p.exists(): return []
    txt = p.read_text(encoding="utf-8",errors="ignore")
    m = re.search(r"(columns?|cols?)\s*[:=]\s*(\[[^\]]+\])", txt, flags=re.I) or re.search(r"\[[^\]]+\]", txt)
    if not m: return []
    block = (m.group(2) if getattr(m,"lastindex",0)>=2 else m.group(0))
    try:
        arr = json.loads(block.replace("'",'"'))
        if isinstance(arr,list) and all(isinstance(x,str) for x in arr): return arr
    except: pass
    return [x.strip().strip("'\"") for x in block.strip("[]").split(",") if x.strip()]

def infer_columns(exp_dir: Path, schema: dict, model):
    cols, ranges0 = [], {}
    if isinstance(schema, dict):
        for k in ["feature_order","columns","features","feature_names","input_columns","input_cols",
                  "fields","variables","predictors","X_columns","x_columns","raw_columns","cols"]:
            v = schema.get(k)
            if isinstance(v,list) and v and all(isinstance(x,str) for x in v):
                cols=v; ranges0=schema.get("ranges") if isinstance(schema.get("ranges"),dict) else {}; break
        if not cols and isinstance(schema.get("ranges"),dict) and schema["ranges"]:
            cols=list(schema["ranges"].keys()); ranges0=schema["ranges"]
    if not cols:
        cols = read_cols_from_readme(exp_dir)
    if not cols and model is not None:
        names = getattr(model,"feature_names_in_", None)
        if names is None and hasattr(model,"named_steps"):
            for step in list(model.named_steps.values())[::-1]:
                names = getattr(step,"feature_names_in_", None)
                if names is not None: break
        if names is not None: cols=[str(x) for x in list(names)]
    return cols, safe_ranges(cols, ranges0)

def apply_interactions(X: pd.DataFrame, inter_map: dict):
    if not inter_map: return X
    X2 = X.copy()
    for name,pair in inter_map.items():
        if isinstance(pair,(list,tuple)) and len(pair)==2:
            a,b = pair
            if a in X2.columns and b in X2.columns:
                X2[name] = pd.to_numeric(X2[a], errors="coerce") * pd.to_numeric(X2[b], errors="coerce")
    return X2

def make_input_row(schema_cols, user_vals: dict, inter_map: dict):
    row = {c: (0.0 if (user_vals.get(c) is None or (isinstance(user_vals.get(c),float) and np.isnan(user_vals[c])) ) else user_vals.get(c)) for c in schema_cols}
    return apply_interactions(pd.DataFrame([row]), inter_map)

def label_and_actions(task: str, y: float):
    if task=="IMP_index":
        if y<=-0.5:  return "冲动偏低（自控较好） / Low impulse", [
            "维持记录日常支出 / Keep expense logging",
            "保留发薪或大额提醒 / Keep payday & large-purchase alerts",
            "适度自我奖励 / Occasional self-reward"]
        if y<=0.5:   return "中等（可控区间） / Moderate", [
            "购物车等待 24 小时再决定 / 24h wait rule",
            "对高频诱因设上限 / Caps for frequent triggers",
            "每周复盘 5 笔最大或后悔支出 / Weekly review top-5 spends"]
        return "冲动偏高（需要干预） / High impulse", [
            "先加购物车 24 小时 / Add-to-cart cooling-off",
            "情绪性消费用低成本替代 / Low-cost emotional substitutes",
            "锁定 1 个高频破功场景设提醒与限额 / Target one trigger with limits"]
    else:
        if y<=-0.3:  return "预算纪律较弱 / Weak discipline", [
            "只跟三类账本：餐饮/通勤/娱乐 / Track 3 core categories",
            "每周 10 分钟对照预算上限 / Weekly 10-min cap check",
            "储蓄改为发薪日自动扣款 / Auto-save on payday"]
        if y<=0.4:   return "预算纪律一般 / Moderate discipline", [
            "为各类支出设软上限 / Soft caps & alerts",
            "坚持一周复盘并给予小奖励 / Weekly review with reward"]
        return "预算纪律良好 / Strong discipline", [
            "维持现状 / Maintain","总结 3 条个人黄金法则 / Write 3 golden rules"]

def try_write_feedback_local(export_dir: str, rows: list):
    """本地运行时写入模型导出目录；云端允许执行但不保证持久。"""
    try:
        if not rows: return None
        p = Path(export_dir)/"feedback_log.csv"
        df = pd.DataFrame(rows)
        if p.exists():
            old = pd.read_csv(p, encoding="utf-8")
            df = pd.concat([old, df], ignore_index=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, encoding="utf-8")
        return str(p)
    except Exception:
        return None


def load_bundle(target: str):
    exp = BASE_DIR/target/"export"
    schema = load_json(exp/"schema.json") or {}
    inter  = load_json(exp/"interactions.json") or {}
    errs   = []
    if not exp.exists(): errs.append(f"{target}：导出目录不存在：{exp}")

    model_path = find_model_path(exp)
    model=None
    if model_path is None: errs.append(f"{target}：未在 {exp} 找到模型文件")
    else:
        model, err = load_model_any(model_path)
        if model is None: errs.append(f"{target}：无法加载模型 {model_path.name} —— {err}")

    cols, ranges = infer_columns(exp, schema, model)
    if not cols: errs.append(f"{target}：无法从 schema/README/模型推断特征列表")

    simple_feats = schema.get("simple_feats")
    if not (isinstance(simple_feats,list) and simple_feats):
        simple_feats = cols[:8] if len(cols)>=8 else cols

    return dict(model=model, schema_cols=cols, simple_feats=simple_feats,
                ranges=ranges, inter_map=inter, errors=errs, export_dir=str(exp))


st.set_page_config(page_title="Finance Assistant", layout="centered")
st.title("Finance Assistant")

# session state 初始化
ss = st.session_state
ss.setdefault("simple_result", None)
ss.setdefault("expert_result", None)
ss.setdefault("feedback_rows", [])

with st.expander("使用说明 / How to use", expanded=True):
    st.markdown("""
本页用于学术演示，将根据你的输入预测两个指数并返回轻量建议。  
• **IMP_index** 越高 → 越容易冲动消费  
• **BD_index** 越高 → 预算纪律越强  
不会采集个人身份信息。提交的体验反馈将汇总为 CSV 供下载。
""")

task = st.sidebar.selectbox("选择要测试的目标 / Choose target", ["IMP_index","BD_index"], index=0)
bundle = load_bundle(task)
if bundle["errors"]:
    st.error("加载资源时遇到问题：\n- " + "\n- ".join(bundle["errors"]))
    st.stop()

model = bundle["model"]
schema_cols = bundle["schema_cols"]
simple_feats = bundle["simple_feats"]
ranges = bundle["ranges"]
inter_map = bundle["inter_map"]
export_dir = bundle["export_dir"]


st.header("简易模式 / Simple Mode")
st.caption("填写少量关键问题；不确定可保持默认值（0≈平均）。")

with st.form("simple_form", clear_on_submit=False):
    user_simple={}
    cols2 = st.columns(2)
    for i,f in enumerate(simple_feats):
        r = ranges.get(f, {"lo":-1,"hi":1.5,"default":0,"step":0.1})
        title = bi(f)
        if f.endswith("_yes"):
            user_simple[f] = cols2[i%2].selectbox(title,[0,1],
                                 index=int(r.get("default",0)>0),
                                 format_func=lambda x:"是/Yes" if x==1 else "否/No",
                                 key=f"simp_{f}")
        else:
            user_simple[f] = cols2[i%2].slider(title, float(r["lo"]), float(r["hi"]),
                                               float(r["default"]), step=float(r["step"]),
                                               key=f"simp_{f}_s")
    submitted = st.form_submit_button("用简易模式预测 / Predict (Simple)")
    if submitted:
        try:
            X = make_input_row(schema_cols, user_simple, inter_map)
            y = float(model.predict(X)[0])
            label, actions = label_and_actions(task, y)
            ss.simple_result = dict(task=task, y=y, label=label, actions=actions,
                                    inputs=user_simple, when=datetime.now().isoformat(timespec="seconds"))
        except Exception as e:
            st.error(f"预测失败：{e}")


if ss.simple_result:
    st.success("简易模式：预测完成 / Done")
    st.metric("预测数值 / Predicted", f"{ss.simple_result['y']:.3f}")
    st.metric("定性判断 / Interpretation", ss.simple_result["label"])
    with st.expander("建议清单 / Action list（Simple）", expanded=True):
        for i,a in enumerate(ss.simple_result["actions"],1):
            st.write(f"{i}. {a}")


st.divider()
st.header("专家模式 / Expert Mode")
st.caption("逐项可控；0 表示平均水平。")

with st.form("expert_form", clear_on_submit=False):
    user_adv={}
    for f in sorted(schema_cols):
        r = ranges.get(f, {"lo":-1,"hi":1.5,"default":0,"step":0.1})
        title = bi(f)
        if f.endswith("_yes"):
            user_adv[f] = st.selectbox(title,[0,1], index=int(r.get("default",0)>0),
                                       format_func=lambda x:"是/Yes" if x==1 else "否/No",
                                       key=f"adv_{f}")
        elif f.endswith("_z"):
            user_adv[f] = st.slider(title, float(r["lo"]), float(r["hi"]),
                                    float(r["default"]), step=float(r["step"]),
                                    key=f"adv_{f}_s")
        else:
            user_adv[f] = st.number_input(title, value=float(r.get("default",0.0)), step=0.1, key=f"adv_{f}_n")
    submitted_e = st.form_submit_button("用专家模式预测 / Predict (Expert)")
    if submitted_e:
        try:
            X = make_input_row(schema_cols, user_adv, inter_map)
            y = float(model.predict(X)[0])
            label, actions = label_and_actions(task, y)
            ss.expert_result = dict(task=task, y=y, label=label, actions=actions,
                                    inputs=user_adv, when=datetime.now().isoformat(timespec="seconds"))
        except Exception as e:
            st.error(f"预测失败：{e}")


if ss.expert_result:
    st.success("专家模式：预测完成 / Done")
    st.metric("预测数值 / Predicted", f"{ss.expert_result['y']:.3f}")
    st.metric("定性判断 / Interpretation", ss.expert_result["label"])
    with st.expander("建议清单 / Action list（Expert）", expanded=True):
        for i,a in enumerate(ss.expert_result["actions"],1):
            st.write(f"{i}. {a}")


st.divider()
st.subheader("你的体验反馈（可选） / Optional UX feedback")


available_ctx = []
if ss.simple_result: available_ctx.append(("Simple", ss.simple_result))
if ss.expert_result: available_ctx.append(("Expert", ss.expert_result))
if not available_ctx:
    st.info("请先完成一次预测，再提交反馈。")
else:
    ctx_names = [f"{name} @ {ctx['when']} [{ctx['task']}]" for name,ctx in available_ctx]
    with st.form("feedback_form", clear_on_submit=True):
        which = st.selectbox("关联结果 / Link to result", list(range(len(ctx_names))),
                             format_func=lambda i: ctx_names[i])
        c1,c2,c3 = st.columns(3)
        easy  = c1.slider("易用 / Ease (1–5)",1,5,4)
        clear = c2.slider("易懂 / Clarity (1–5)",1,5,4)
        trust = c3.slider("可信 / Trust (1–5)",1,5,4)
        comment = st.text_input("其他建议或吐槽 / Other comments")
        sent = st.form_submit_button("提交反馈 / Submit feedback")
        if sent:
            mode_name, ctx = available_ctx[which]
            row = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mode": mode_name,
                "task": ctx["task"],
                "y_pred": ctx["y"],
                "label": ctx["label"],
                "easy": int(easy),
                "understand": int(clear),
                "trust": int(trust),
                "comment": comment,
                "inputs_json": json.dumps(ctx["inputs"], ensure_ascii=False),
            }
            ss.feedback_rows.append(row)
            
            path_written = try_write_feedback_local(export_dir, ss.feedback_rows)
            msg = f"已追加 {len(ss.feedback_rows)} 条；" + (f"已写入：{path_written}" if path_written else "云端请用下面按钮下载 CSV")
            st.success(msg)


if ss.feedback_rows:
    st.download_button(
        "下载本页反馈.csv / Download feedback CSV",
        data=pd.DataFrame(ss.feedback_rows).to_csv(index=False).encode("utf-8"),
        file_name="feedback_log.csv",
        mime="text/csv"
    )
    if st.button("清空本页反馈（仅当前会话） / Clear feedback (session)"):
        ss.feedback_rows = []


with st.expander("其它 / Utilities"):
    if st.button("清空预测结果 / Reset results"):
        ss.simple_result = None
        ss.expert_result = None
        st.experimental_rerun()

st.caption("© Junze Sun — WBS MSc Dissertation Demo · For research demo only; not financial advice.")
