# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 20:31:07 2025

@author: Eden
"""



import json, re, io
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Finance Assistant", layout="centered")

BASE_DIR = Path(__file__).parent / "modeling_plus"

ZH = {
    "IMP_index":"冲动消费指数（目标）", "BD_index":"预算纪律指数（目标）",
    "AFF_index":"情绪/社交动因综合指数", "BD_F1":"预算纪律F1（执行/遵守）",
    "SAV_F1":"储蓄F1（基础习惯）", "SAV_F2":"储蓄F2（目标/计划）",
    "INV_breadth_z":"投资广度（标准化）", "EF1_score":"执行功能EF1（计划/注意）",
    "EF4_score":"执行功能EF4（规则/切换）", "EF7_score":"执行功能EF7（自控/延迟满足）",
    "EF9_score":"执行功能EF9（情绪自我调节）",
    "Q7_yes":"是否设定每月预算（Q7）", "Q32_yes":"是否有过投资行为（Q32）"
}
EN = {
    "IMP_index":"Impulse Spending (target)","BD_index":"Budget Discipline (target)",
    "AFF_index":"Affect/Social Drivers Index","BD_F1":"Budget Discipline F1",
    "SAV_F1":"Saving Factor F1","SAV_F2":"Saving Factor F2",
    "INV_breadth_z":"Investment Breadth (z)","EF1_score":"Executive Function EF1",
    "EF4_score":"Executive Function EF4","EF7_score":"Executive Function EF7",
    "EF9_score":"Executive Function EF9",
    "Q7_yes":"Set monthly budget (Q7)","Q32_yes":"Ever invested (Q32)"
}

def zh(n): 
    if n.startswith("INT__"):
        p=n.split("__"); 
        if len(p)>=4 and p[2]=="x":
            a,b=p[1],p[3]; 
            return f"交互：{ZH.get(a,a)} × {ZH.get(b,b)}"
    return ZH.get(n,n)
def en(n): 
    if n.startswith("INT__"):
        p=n.split("__"); 
        if len(p)>=4 and p[2]=="x":
            a,b=p[1],p[3]; 
            return f"Interaction: {EN.get(a,a)} × {EN.get(b,b)}"
    return EN.get(n,n)
def bi(n): return f"{zh(n)} / {en(n)}（{n}）"

@st.cache_data(show_spinner=False)
def load_json(p: Path):
    if not p.exists(): return None
    for enc in ("utf-8","utf-8-sig"):
        try: return json.loads(p.read_text(encoding=enc))
        except Exception: pass
    return None

def find_model_path(exp_dir: Path):
    for name in ["model.pkl","model.joblib","pipeline.pkl","clf.pkl","estimator.pkl"]:
        p=exp_dir/name
        if p.exists(): return p
    for p in list(exp_dir.glob("*.pkl"))+list(exp_dir.glob("*.joblib")):
        return p
    return None

@st.cache_resource(show_spinner=False)
def load_model_any(p: Path):
    try: return joblib.load(p), None
    except Exception as e: return None, f"{type(e).__name__}: {e}"

def safe_ranges(cols, ranges):
    if not isinstance(ranges, dict): ranges={}
    def guess(name):
        if name.endswith("_yes"): return {"lo":0,"hi":1,"default":0,"step":1}
        if name.endswith("_z"): return {"lo":-2.0,"hi":2.0,"default":0.0,"step":0.1}
        return {"lo":-1.5,"hi":1.5,"default":0.0,"step":0.1}
    out={}
    for f in cols: out[f]=ranges.get(f,guess(f))
    return out

def read_columns_from_readme(exp_dir: Path):
    p=exp_dir/"README_export.txt"
    if not p.exists(): return []
    try: txt=p.read_text(encoding="utf-8",errors="ignore")
    except Exception: return []
    m=re.search(r"(columns?|cols?)\s*[:=]\s*(\[[^\]]+\])",txt,re.I)
    if not m: m=re.search(r"\[[^\]]+\]",txt)
    if not m: return []
    block=m.group(2) if m.lastindex and m.lastindex>=2 else m.group(0)
    try:
        arr=json.loads(block.replace("'","\""))
        if isinstance(arr,list) and all(isinstance(x,str) for x in arr): return arr
    except Exception: pass
    items=[x.strip().strip("'\"") for x in block.strip("[]").split(",")]
    return [x for x in items if x]

def infer_columns(exp_dir: Path, schema: dict, model):
    cols=[]; ranges0={}
    if isinstance(schema, dict):
        for k in ["feature_order","columns","cols","features","feature_names","featureNames",
                  "input_columns","input_cols","fields","variables","predictors",
                  "X_columns","x_columns","raw_columns"]:
            v=schema.get(k)
            if isinstance(v,list) and all(isinstance(x,str) for x in v) and len(v)>0:
                cols=v; ranges0=schema.get("ranges") if isinstance(schema.get("ranges"),dict) else {}
                break
        if not cols and isinstance(schema.get("ranges"),dict) and len(schema["ranges"])>0:
            cols=list(schema["ranges"].keys()); ranges0=schema["ranges"]
    if not cols: cols=read_columns_from_readme(exp_dir)
    if not cols and model is not None:
        names=getattr(model,"feature_names_in_",None)
        if names is None and hasattr(model,"named_steps"):
            for step in list(model.named_steps.values())[::-1]:
                names=getattr(step,"feature_names_in_",None)
                if names is not None: break
        if names is not None: cols=[str(x) for x in list(names)]
    return cols, safe_ranges(cols, ranges0)

def load_bundle(target: str):
    exp_dir = BASE_DIR / target / "export"
    schema = load_json(exp_dir / "schema.json") or {}
    inter_map = load_json(exp_dir / "interactions.json") or {}
    errors=[]
    if not exp_dir.exists(): errors.append(f"{target}：导出目录不存在：{exp_dir}")
    model_path=find_model_path(exp_dir)
    model=None
    if model_path is None: errors.append(f"{target}：未找到模型文件")
    else:
        model,err=load_model_any(model_path)
        if model is None: errors.append(f"{target}：无法加载模型 {model_path.name} —— {err}")
    cols,ranges=infer_columns(exp_dir, schema, model)
    if not cols: errors.append(f"{target}：无法从 schema/README/模型中推断特征列表")
    simple_feats=schema.get("simple_feats")
    if not isinstance(simple_feats,list) or not simple_feats:
        guess=[x for x in cols if x in ["AFF_index","BD_F1","EF1_score","EF4_score","EF7_score","EF9_score",
                                        "SAV_F1","SAV_F2","INV_breadth_z","Q7_yes","Q32_yes"]]
        simple_feats = guess[:8] if len(guess)>=6 else cols[:8]
    return {"model":model,"schema_cols":cols,"simple_feats":simple_feats,
            "ranges":ranges,"inter_map":inter_map,"errors":errors}

def apply_interactions(X: pd.DataFrame, inter_map: dict):
    if not inter_map: return X
    X2=X.copy()
    for name,pair in inter_map.items():
        if isinstance(pair,(list,tuple)) and len(pair)==2:
            a,b=pair
            if a in X2.columns and b in X2.columns:
                X2[name]=pd.to_numeric(X2[a],errors="coerce")*pd.to_numeric(X2[b],errors="coerce")
    return X2

def make_input_row(schema_cols, user_vals: dict, inter_map: dict):
    row={c: user_vals.get(c, 0.0) if user_vals.get(c,0.0) is not None else 0.0 for c in schema_cols}
    X=pd.DataFrame([row])
    return apply_interactions(X, inter_map)

def label_and_actions(task: str, y: float):
    if task=="IMP_index":
        if y<=-0.30:
            return "冲动偏低 / Low", ["维持记录日常支出","保留发薪/大额提醒","适度自我奖励"]
        elif y<=0.30:
            return "中等 / Moderate", ["购物车等待24小时再决定","对高频诱因设上限","每周复盘5笔最大或后悔支出"]
        else:
            return "冲动偏高 / High", ["加购物车冷静24h","情绪性消费用低成本替代","锁定1个高频破功场景设提醒与限额"]
    else:
        if y<=-0.20:
            return "预算纪律较弱 / Weak", ["只跟三类账本：餐饮/通勤/娱乐","每周10分钟对照预算上限","发薪日自动扣款储蓄"]
        elif y<=0.30:
            return "预算纪律一般 / Moderate", ["各类支出设软上限+提醒","坚持一周复盘并给予小奖励"]
        else:
            return "预算纪律良好 / Strong", ["维持现状","总结3条个人黄金法则"]

def ensure_state():
    if "records" not in st.session_state: st.session_state.records=[]
    if "counter" not in st.session_state: st.session_state.counter=0

def add_result_row(task, mode, y_pred, label, inputs):
    ensure_state()
    st.session_state.counter += 1
    rid = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{st.session_state.counter}"
    row = {
        "kind":"result","row_id":rid,"ts":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task":task,"mode":mode,"y_pred":float(y_pred),"label":label,
        "inputs_json":json.dumps(inputs, ensure_ascii=False)
    }
    st.session_state.records.append(row)
    return rid

def add_feedback_row(ref_row_id, easy, clarity, trust, comment):
    ensure_state()
    st.session_state.counter += 1
    row = {
        "kind":"feedback","row_id":f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{st.session_state.counter}",
        "ts":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ref_id":ref_row_id,"easy":int(easy),"clarity":int(clarity),"trust":int(trust),
        "comment":str(comment or "")
    }
    st.session_state.records.append(row)
    return row["row_id"]

def records_dataframe():
    ensure_state()
    if not st.session_state.records: return pd.DataFrame()
    return pd.DataFrame(st.session_state.records)

st.title("Finance Assistant")
with st.expander("使用说明 / How to use", expanded=True):
    st.markdown("""
本页用于学术演示。选择目标后，可用**简易模式**或**专家模式**进行预测；每次点击「Predict」会把一条结果写入页面会话的日志。  
在页面底部可「下载本次会话所有结果（CSV）」。  
在「你的体验反馈」区域可将反馈与某一条结果关联，并写入同一个 CSV（kind=feedback，ref_id 指向结果行）。
""")

task = st.sidebar.selectbox("选择要测试的目标 / Choose target", ["IMP_index","BD_index"], index=0)
bundle = load_bundle(task)
if bundle["errors"]:
    st.error("加载资源时遇到问题：\n- " + "\n- ".join(bundle["errors"]))
    st.stop()

model=bundle["model"]; schema_cols=bundle["schema_cols"]
simple_feats=bundle["simple_feats"]; ranges=bundle["ranges"]; inter_map=bundle["inter_map"]

st.header("简易模式 / Simple Mode")
st.caption("填写少量关键问题；0≈平均。")
user_simple={}
cols2=st.columns(2)
for i,f in enumerate(simple_feats):
    r=ranges.get(f,{"lo":-1.0,"hi":1.5,"default":0.0,"step":0.1})
    title=bi(f)
    if f.endswith("_yes"):
        user_simple[f]=cols2[i%2].selectbox(title,[0,1],index=int(r.get("default",0)>0),
                                           format_func=lambda x:"是/Yes" if x==1 else "否/No",key=f"simp_{f}")
    else:
        user_simple[f]=cols2[i%2].slider(title,float(r["lo"]),float(r["hi"]),float(r["default"]),
                                         step=float(r["step"]),key=f"simp_{f}")

if st.button("用简易模式预测 / Predict (Simple)", type="primary"):
    X=make_input_row(schema_cols,user_simple,inter_map)
    try: y=float(model.predict(X)[0])
    except Exception as e:
        st.error(f"模型预测失败：{e}"); st.stop()
    label, actions = label_and_actions(task,y)
    rid=add_result_row(task,"simple",y,label,user_simple)
    st.success("简易模式：预测完成 / Done")
    c1,c2=st.columns(2)
    c1.metric("预测数值 / Predicted", f"{y:.3f}")
    c2.metric("定性判断 / Interpretation", label)
    with st.expander("建议清单 / Action list (Simple)", expanded=True):
        for i,a in enumerate(actions,1): st.write(f"{i}. {a}")
    st.info(f"已记录到会话日志（row_id={rid)})")

st.divider()
st.header("专家模式 / Expert Mode")
st.caption("逐项可控；0≈平均。")
user_adv={}
for f in schema_cols:
    r=ranges.get(f,{"lo":-1.0,"hi":1.5,"default":0.0,"step":0.1})
    title=bi(f)
    if f.endswith("_yes"):
        user_adv[f]=st.selectbox(title,[0,1],index=int(r.get("default",0)>0),
                                 format_func=lambda x:"是/Yes" if x==1 else "否/No", key=f"adv_{f}")
    elif f.endswith("_z"):
        user_adv[f]=st.slider(title,float(r["lo"]),float(r["hi"]),float(r["default"]),
                              step=float(r["step"]), key=f"adv_{f}")
    else:
        user_adv[f]=st.number_input(title, value=float(r.get("default",0.0)), step=0.1, key=f"adv_{f}")

if st.button("用专家模式预测 / Predict (Expert)"):
    X=make_input_row(schema_cols,user_adv,inter_map)
    try: y=float(model.predict(X)[0])
    except Exception as e:
        st.error(f"模型预测失败：{e}"); st.stop()
    label, actions = label_and_actions(task,y)
    rid=add_result_row(task,"expert",y,label,user_adv)
    st.success("专家模式：预测完成 / Done")
    c1,c2=st.columns(2)
    c1.metric("预测数值 / Predicted", f"{y:.3f}")
    c2.metric("定性判断 / Interpretation", label)
    with st.expander("建议清单 / Action list (Expert)", expanded=True):
        for i,a in enumerate(actions,1): st.write(f"{i}. {a}")
    st.info(f"已记录到会话日志（row_id={rid)})")

st.divider()
st.subheader("你的体验反馈（可选） / Optional UX feedback")

ensure_state()
result_rows=[r for r in st.session_state.records if r.get("kind")=="result"]
pretty=lambda r: f"{r['mode'].capitalize()} @ {r['ts']} [{r['task']}]  id={r['row_id']}"
if not result_rows:
    st.info("请先完成一次预测，再提交反馈。")
else:
    default_idx=len(result_rows)-1
    ref_choice=st.selectbox("关联结果 / Link to result", options=list(range(len(result_rows))),
                            format_func=lambda i: pretty(result_rows[i]), index=default_idx)
    c1,c2,c3=st.columns(3)
    easy=c1.slider("易用 / Ease (1–5)",1,5,4)
    clarity=c2.slider("易懂 / Clarity (1–5)",1,5,4)
    trust=c3.slider("可信 / Trust (1–5)",1,5,4)
    comment=st.text_input("其他建议或吐槽 / Other comments")
    if st.button("提交反馈 / Submit feedback"):
        ref_id=result_rows[ref_choice]["row_id"]
        fb_id=add_feedback_row(ref_id,easy,clarity,trust,comment)
        st.success(f"反馈已记录（feedback_id={fb_id}, ref_id={ref_id}）")

st.divider()
with st.expander("其它 / Utilities", expanded=False):
    df=records_dataframe()
    st.write(f"本次会话已记录：{len(df)} 行")
    if not df.empty:
        csv=df.to_csv(index=False).encode("utf-8")
        st.download_button("下载本次会话所有结果（CSV）", data=csv, file_name="session_results.csv",
                           mime="text/csv")
    if st.button("清空本次会话记录 / Clear session log"):
        st.session_state.records=[]; st.session_state.counter=0
        st.success("已清空")
