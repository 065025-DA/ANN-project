import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="UPI Fraud ANN Dashboard", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#0a0a0f;color:#e8e8f0;}
.stApp{background:#0a0a0f;}
section[data-testid="stSidebar"]{background:#12121a!important;border-right:1px solid #2a2a40;}
.mc{background:#1a1a26;border:1px solid #2a2a40;border-radius:12px;padding:18px;
    text-align:center;position:relative;overflow:hidden;margin-bottom:6px;}
.mc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
            background:linear-gradient(90deg,#7c3aff,#00ff88);}
.mv{font-size:2rem;font-weight:800;color:#00ff88;font-family:'Space Mono',monospace;line-height:1.1;}
.ml{font-size:0.68rem;color:#888899;text-transform:uppercase;letter-spacing:2px;margin-top:5px;}
.hero{background:linear-gradient(135deg,#0a0a0f,#1a0a2e,#0a1a0f);border:1px solid #2a2a40;
      border-radius:16px;padding:28px 36px;margin-bottom:24px;}
.ht{font-size:1.9rem;font-weight:800;background:linear-gradient(90deg,#00ff88,#7c3aff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;}
.hs{color:#888899;font-family:'Space Mono',monospace;font-size:0.7rem;margin-top:8px;letter-spacing:1px;}
.st-title{font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:4px;
          color:#888899;border-bottom:1px solid #2a2a40;padding-bottom:8px;margin:22px 0 14px;}
.badge{display:inline-block;background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.25);
       color:#00ff88;font-family:'Space Mono',monospace;font-size:0.65rem;
       padding:2px 9px;border-radius:20px;margin:2px;}
.badge-r{background:rgba(255,51,102,0.08);border-color:rgba(255,51,102,0.25);color:#ff3366;}
.badge-p{background:rgba(124,58,255,0.08);border-color:rgba(124,58,255,0.25);color:#7c3aff;}
.badge-y{background:rgba(255,170,0,0.08);border-color:rgba(255,170,0,0.25);color:#ffaa00;}
.tlog{background:#0a0a0f;border:1px solid #2a2a40;border-radius:8px;padding:12px 16px;
      font-family:'Space Mono',monospace;font-size:0.73rem;color:#00ff88;line-height:1.8;}
.wbox{background:rgba(255,51,102,0.05);border:1px solid rgba(255,51,102,0.2);border-radius:8px;
      padding:16px;font-size:0.8rem;color:#ff8099;font-family:'Space Mono',monospace;}
.ibox{background:rgba(255,170,0,0.05);border:1px solid rgba(255,170,0,0.25);border-radius:10px;
      padding:14px 18px;font-size:0.8rem;color:#ffcc66;font-family:'Space Mono',monospace;
      margin-bottom:14px;line-height:1.7;}
.stButton>button{background:linear-gradient(135deg,#7c3aff,#00ff88)!important;color:#000!important;
  font-family:'Syne',sans-serif!important;font-weight:700!important;border:none!important;
  border-radius:8px!important;text-transform:uppercase!important;letter-spacing:1px!important;}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent!important;}
.stTabs [data-baseweb="tab"]{background:#1a1a26!important;border:1px solid #2a2a40!important;
  border-radius:8px!important;color:#888899!important;font-weight:700;font-size:0.75rem;
  text-transform:uppercase;letter-spacing:1px;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(124,58,255,0.2),
  rgba(0,255,136,0.1))!important;border-color:#00ff88!important;color:#00ff88!important;}
</style>
""", unsafe_allow_html=True)

CL = dict(paper_bgcolor='#1a1a26', plot_bgcolor='#1a1a26', font_color='#e8e8f0',
          font_family='Syne',
          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8e8f0', size=10)),
          margin=dict(t=40, b=20, l=20, r=20))

# ── Load 50,000 rows ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("upi_transactions_2024.csv", nrows=50000)
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(' ', '_').str.replace('(', '').str.replace(')', ''))
    return df

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:16px 0 20px;">
      <div style="font-size:2rem;">🧠</div>
      <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;
                  background:linear-gradient(90deg,#00ff88,#7c3aff);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">ANN Tuner</div>
      <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#888899;margin-top:4px;">
        HYPERPARAMETER CONTROL</div></div>""", unsafe_allow_html=True)

    st.markdown("**🏗 Architecture**")
    n_layers    = st.slider("Hidden Layers", 1, 5, 3)
    defaults    = [128, 64, 32, 16, 8]
    layer_sizes = [st.slider(f"Layer {i+1} Neurons", 8, 256, defaults[i], step=8, key=f"l{i}")
                   for i in range(n_layers)]

    st.markdown("---")
    st.markdown("**⚙ Training**")
    activation  = st.selectbox("Activation",    ["relu", "tanh", "sigmoid", "elu", "selu"])
    optimizer   = st.selectbox("Optimizer",     ["adam", "sgd", "rmsprop", "adamax"])
    lr          = st.select_slider("Learning Rate",
                    [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], value=0.001)
    epochs      = st.slider("Epochs", 5, 100, 40, step=5)
    batch_size  = st.select_slider("Batch Size", [16, 32, 64, 128, 256], value=32)
    dropout     = st.slider("Dropout", 0.0, 0.7, 0.4, step=0.05)

    st.markdown("---")
    st.markdown("**🔧 Imbalance Handling**")
    use_smote   = st.checkbox("Apply SMOTE", value=True)
    
    # Class weight option — important for severe imbalance
    use_class_weight = st.checkbox("Use Class Weights", value=True)
    
    # Decision threshold tuning — crucial for 0.15% fraud rate
    threshold   = st.slider("Decision Threshold", 0.1, 0.9, 0.3, step=0.05,
                             help="Lower threshold = catch more fraud (higher recall). Default 0.5 misses most fraud at 0.15% rate.")
    
    test_split  = st.slider("Test Split %", 10, 40, 20)
    st.markdown("---")
    train_btn   = st.button("🚀 TRAIN MODEL", use_container_width=True)

    fraud_count = int(df['fraud_flag'].sum())
    legit_count = len(df) - fraud_count
    st.markdown(f"""<div style="margin-top:14px;padding:10px;background:#0a0a0f;border:1px solid #2a2a40;
    border-radius:8px;font-family:'Space Mono',monospace;font-size:0.6rem;color:#888899;">
    ⚡ 50,000 rows loaded<br>
    🎯 Target: fraud_flag<br>
    🔴 Fraud: {fraud_count} ({fraud_count/len(df)*100:.2f}%)<br>
    🟢 Legit: {legit_count:,}<br>
    ⚖ Ratio: 1 : {legit_count//max(fraud_count,1)}<br>
    📐 Features: 8 selected</div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""<div class="hero">
  <p class="ht">UPI Fraud Detection<br>Neural Network Dashboard</p>
  <p class="hs">◈  ANN · HYPERPARAMETER TUNING · LIVE TRAINING · SEVERE IMBALANCE (0.15%) · SMOTE · CLASS WEIGHTS</p>
</div>""", unsafe_allow_html=True)

# Imbalance warning banner
fraud_pct = df['fraud_flag'].mean() * 100
st.markdown(f"""<div class="ibox">
⚠️ <b>SEVERE CLASS IMBALANCE DETECTED</b> — Fraud rate: <b>{fraud_pct:.3f}%</b>
({int(df['fraud_flag'].sum())} fraud out of {len(df):,} transactions)<br>
📌 Standard accuracy is <b>misleading</b> here — focus on <b>ROC-AUC, Recall & F1-Fraud</b>.<br>
📌 SMOTE + Class Weights + <b>lowered decision threshold ({threshold})</b> are enabled to handle this.
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA & Overview", "🧠 Train ANN", "📈 Results & Metrics", "🔍 Live Predictor"])

# ══ TAB 1 — EDA ══════════════════════════════════════════════════════════════
with tab1:
    total   = len(df)
    fraud   = int(df['fraud_flag'].sum())
    legit   = total - fraud
    fpct    = fraud / total * 100
    avg_amt = df['amount_inr'].mean()
    fraud_amt = df[df['fraud_flag']==1]['amount_inr'].mean()

    cols = st.columns(6)
    for col, (lab, val, clr) in zip(cols, [
        ("Total Txns",    f"{total:,}",          "#e8e8f0"),
        ("Fraud Cases",   f"{fraud}",             "#ff3366"),
        ("Legit Cases",   f"{legit:,}",           "#00ff88"),
        ("Fraud Rate",    f"{fpct:.3f}%",         "#ff3366"),
        ("Avg Amount",    f"₹{avg_amt:,.0f}",     "#ffaa00"),
        ("Avg Fraud Amt", f"₹{fraud_amt:,.0f}",   "#7c3aff"),
    ]):
        with col:
            st.markdown(f'<div class="mc"><div class="mv" style="color:{clr};">{val}</div>'
                        f'<div class="ml">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("<div class='st-title'>Class Imbalance — The Core Challenge</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        # Log scale bar to show the extreme imbalance visually
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Legitimate', 'Fraud'], y=[legit, fraud],
            marker_color=['#00ff88', '#ff3366'],
            text=[f"{legit:,}", f"{fraud}"],
            textposition='outside',
            textfont=dict(family='Space Mono', size=12, color='white')))
        fig.update_layout(**CL, height=300, showlegend=False,
            title=dict(text=f"Class Distribution (Log Scale) — {fraud} fraud vs {legit:,} legit",
                       font=dict(color='#888899', size=12)),
            yaxis=dict(type='log', gridcolor='#2a2a40', title='Count (log)'),
            xaxis=dict(gridcolor='#2a2a40'))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = go.Figure(go.Pie(
            labels=['Legitimate', 'Fraud'], values=[legit, fraud],
            hole=0.65, marker_colors=['#00ff88', '#ff3366'],
            textinfo='label+percent',
            textfont=dict(family='Space Mono', size=11)))
        fig2.add_annotation(text=f"<b>{fpct:.3f}%</b><br>Fraud",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color='#ff3366', family='Space Mono'))
        fig2.update_layout(**CL, height=300,
            title=dict(text="Extreme Imbalance — Why SMOTE & Threshold Tuning Matter",
                       font=dict(color='#888899', size=12)))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='st-title'>Fraud Pattern Analysis</div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        fig3 = go.Figure()
        for flag, nm, clr in [(0,'Legit','rgba(0,255,136,0.5)'), (1,'Fraud','rgba(255,51,102,0.8)')]:
            fig3.add_trace(go.Histogram(x=df[df['fraud_flag']==flag]['amount_inr'],
                name=nm, nbinsx=50, marker_color=clr, opacity=0.85))
        fig3.update_layout(**CL, barmode='overlay', height=290,
            title=dict(text="Amount Distribution by Class", font=dict(color='#888899', size=13)),
            xaxis=dict(title='INR', gridcolor='#2a2a40'),
            yaxis=dict(title='Count', gridcolor='#2a2a40'))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fh = df.groupby('hour_of_day')['fraud_flag'].agg(['sum','count']).reset_index()
        fh['rate'] = fh['sum'] / fh['count'] * 100
        fig4 = go.Figure(go.Scatter(x=fh['hour_of_day'], y=fh['rate'],
            mode='lines+markers', fill='tozeroy',
            fillcolor='rgba(124,58,255,0.12)',
            line=dict(color='#7c3aff', width=2.5),
            marker=dict(color='#00ff88', size=6)))
        fig4.update_layout(**CL, height=290,
            title=dict(text="Fraud Rate by Hour of Day", font=dict(color='#888899', size=13)),
            xaxis=dict(title='Hour', gridcolor='#2a2a40', dtick=2),
            yaxis=dict(title='Fraud %', gridcolor='#2a2a40'))
        st.plotly_chart(fig4, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        fbt = df.groupby('transaction_type')['fraud_flag'].agg(['sum','count']).reset_index()
        fbt['rate'] = fbt['sum'] / fbt['count'] * 100
        fbt = fbt.sort_values('rate')
        fig5 = go.Figure(go.Bar(y=fbt['transaction_type'], x=fbt['rate'], orientation='h',
            marker=dict(color=fbt['rate'], colorscale=[[0,'#7c3aff'],[1,'#ff3366']], showscale=False),
            text=[f"{v:.2f}%" for v in fbt['rate']], textposition='outside',
            textfont=dict(family='Space Mono', size=10)))
        fig5.update_layout(**CL, height=290,
            title=dict(text="Fraud % by Txn Type", font=dict(color='#888899', size=13)),
            xaxis=dict(title='Fraud %', gridcolor='#2a2a40'), yaxis=dict(gridcolor='#2a2a40'))
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        sf = df.groupby('sender_state')['fraud_flag'].agg(['sum','count']).reset_index()
        sf['rate'] = sf['sum'] / sf['count'] * 100
        sf = sf.sort_values('rate', ascending=False).head(10)
        fig6 = px.bar(sf, x='sender_state', y='rate', color='rate',
            color_continuous_scale=['#7c3aff','#ff3366'])
        fig6.update_layout(**CL, height=290, coloraxis_showscale=False,
            title=dict(text="Top 10 States — Fraud Rate", font=dict(color='#888899', size=13)),
            xaxis=dict(tickangle=30, gridcolor='#2a2a40'),
            yaxis=dict(title='Fraud %', gridcolor='#2a2a40'))
        st.plotly_chart(fig6, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        bf = df.groupby('sender_bank')['fraud_flag'].agg(['sum','count']).reset_index()
        bf['rate'] = bf['sum'] / bf['count'] * 100
        bf = bf.sort_values('rate', ascending=False)
        fig7 = px.bar(bf, x='sender_bank', y='rate', color='rate',
            color_continuous_scale=['#1a1a26','#7c3aff','#ff3366'])
        fig7.update_layout(**CL, height=290, coloraxis_showscale=False,
            title=dict(text="Fraud % by Bank", font=dict(color='#888899', size=13)),
            xaxis=dict(tickangle=30, gridcolor='#2a2a40'),
            yaxis=dict(title='Fraud %', gridcolor='#2a2a40'))
        st.plotly_chart(fig7, use_container_width=True)

    with c8:
        cat_f = df.groupby('merchant_category')['fraud_flag'].agg(['sum','count']).reset_index()
        cat_f['rate'] = cat_f['sum'] / cat_f['count'] * 100
        cat_f = cat_f.sort_values('rate', ascending=False)
        fig8 = px.bar(cat_f, x='merchant_category', y='rate', color='rate',
            color_continuous_scale=['#7c3aff','#ff3366'])
        fig8.update_layout(**CL, height=290, coloraxis_showscale=False,
            title=dict(text="Fraud % by Merchant Category", font=dict(color='#888899', size=13)),
            xaxis=dict(tickangle=30, gridcolor='#2a2a40'),
            yaxis=dict(title='Fraud %', gridcolor='#2a2a40'))
        st.plotly_chart(fig8, use_container_width=True)

    # Weekend vs Weekday
    c9, c10 = st.columns(2)
    with c9:
        wk = df.groupby('is_weekend')['fraud_flag'].agg(['sum','count']).reset_index()
        wk['rate'] = wk['sum'] / wk['count'] * 100
        wk['label'] = wk['is_weekend'].map({0:'Weekday', 1:'Weekend'})
        fig9 = px.bar(wk, x='label', y='rate', color='label',
            color_discrete_map={'Weekday':'#7c3aff','Weekend':'#ff3366'})
        fig9.update_layout(**CL, height=270, showlegend=False,
            title=dict(text="Fraud Rate: Weekday vs Weekend", font=dict(color='#888899', size=13)),
            xaxis=dict(gridcolor='#2a2a40'), yaxis=dict(title='Fraud %', gridcolor='#2a2a40'))
        st.plotly_chart(fig9, use_container_width=True)
    with c10:
        nm = df.groupby('network_type')['fraud_flag'].agg(['sum','count']).reset_index()
        nm['rate'] = nm['sum'] / nm['count'] * 100
        fig10 = px.pie(nm, names='network_type', values='rate',
            color_discrete_sequence=['#7c3aff','#ff3366','#00ff88','#ffaa00'])
        fig10.update_layout(**CL, height=270,
            title=dict(text="Fraud Share by Network Type", font=dict(color='#888899', size=13)))
        st.plotly_chart(fig10, use_container_width=True)

# ══ TAB 2 — TRAIN ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='st-title'>Architecture Preview</div>", unsafe_allow_html=True)

    all_l   = [8] + layer_sizes + [1]
    lnames  = ['Input\n(8 feats)'] + [f'Dense({s})\n{activation}' for s in layer_sizes] + ['Output\nSigmoid']
    max_n   = max(all_l)
    fig_a   = go.Figure()

    for li, (n, name) in enumerate(zip(all_l, lnames)):
        clr = '#00ff88' if li == 0 else ('#ff3366' if li == len(all_l)-1 else '#7c3aff')
        sp  = max_n / (n + 1)
        ys  = [(j+1)*sp - max_n/2 for j in range(n)]
        fig_a.add_trace(go.Scatter(x=[li]*n, y=ys, mode='markers',
            marker=dict(size=max(6, min(20, 100//n)), color=clr,
                        line=dict(color='white', width=0.8), opacity=0.85),
            showlegend=False, hoverinfo='skip'))
        if li < len(all_l)-1:
            n2 = all_l[li+1]; sp2 = max_n/(n2+1)
            for a in range(min(n, 6)):
                y1 = (a+1)*sp - max_n/2
                for b in range(min(n2, 6)):
                    y2 = (b+1)*sp2 - max_n/2
                    fig_a.add_trace(go.Scatter(x=[li, li+1], y=[y1, y2], mode='lines',
                        line=dict(color='rgba(124,58,255,0.1)', width=0.4),
                        showlegend=False, hoverinfo='skip'))
        fig_a.add_annotation(x=li, y=-max_n/2-2.5, text=name.replace('\n','<br>'),
            showarrow=False, font=dict(size=8, color='#888899', family='Space Mono'), align='center')

    fig_a.update_layout(paper_bgcolor='#1a1a26', plot_bgcolor='#1a1a26',
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   range=[-0.6, len(all_l)-0.4]),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=350, margin=dict(t=10, b=75, l=10, r=10))
    st.plotly_chart(fig_a, use_container_width=True)

    arch_str = " → ".join(str(s) for s in all_l)
    smote_b  = '<span class="badge">✅ SMOTE</span>' if use_smote else '<span class="badge badge-r">✗ SMOTE OFF</span>'
    cw_b     = '<span class="badge">✅ Class Weights</span>' if use_class_weight else '<span class="badge badge-r">✗ No Class Weights</span>'
    st.markdown(f"""<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:16px;">
      <span class="badge badge-p">Arch: [{arch_str}]</span>
      <span class="badge">Act: {activation}</span>
      <span class="badge">Opt: {optimizer} lr={lr}</span>
      <span class="badge">Dropout: {dropout}</span>
      <span class="badge">Batch:{batch_size} Epochs:{epochs}</span>
      <span class="badge badge-y">Threshold: {threshold}</span>
      {smote_b}{cw_b}</div>""", unsafe_allow_html=True)

    # Why threshold matters — explainer
    st.markdown(f"""<div class="ibox">
    📌 <b>Why threshold = {threshold} instead of 0.5?</b><br>
    With only {int(df['fraud_flag'].sum())} fraud cases in 50,000 rows ({fraud_pct:.3f}%), the model
    rarely predicts values above 0.5 for fraud. Lowering the threshold to {threshold} means:
    any transaction the model gives ≥{threshold} fraud probability gets flagged —
    catching far more real fraud cases (higher Recall) at the cost of some false alarms.
    This is standard practice in fraud detection systems.
    </div>""", unsafe_allow_html=True)

    if train_btn:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.utils.class_weight import compute_class_weight
        from sklearn.metrics import (classification_report, confusion_matrix,
            roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers as kl, regularizers

        pb   = st.progress(0)
        stat = st.empty()
        stat.info("⏳ Preprocessing 50,000 rows…")

        df_m = df.copy()
        cat_cols = ['transaction_type','merchant_category','transaction_status','sender_state',
                    'sender_bank','receiver_bank','device_type','network_type',
                    'sender_age_group','receiver_age_group','day_of_week']
        le_map = {}
        for c in cat_cols:
            if c in df_m.columns:
                le = LabelEncoder()
                df_m[c] = le.fit_transform(df_m[c].astype(str))
                le_map[c] = le

        feat_cols = ['transaction_type','merchant_category','amount_inr','sender_age_group',
                     'receiver_age_group','hour_of_day','is_weekend','device_type']
        X = df_m[feat_cols].values
        y = df_m['fraud_flag'].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_split/100, random_state=42, stratify=y)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te  = sc.transform(X_te)
        pb.progress(15)

        # ── SMOTE ────────────────────────────────────────────────────
        smote_ok = False
        if use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                # With very few fraud cases, use k_neighbors=1 to be safe
                n_fraud_tr = int(y_tr.sum())
                k = min(3, n_fraud_tr - 1) if n_fraud_tr > 1 else 1
                sm = SMOTE(random_state=42, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                smote_ok = True
                stat.success(f"✅ SMOTE done — {len(X_tr):,} training samples "
                             f"({int(y_tr.sum()):,} fraud, {int((y_tr==0).sum()):,} legit)")
            except ImportError:
                stat.warning("⚠️ imbalanced-learn not found. Run: pip install imbalanced-learn")
            except Exception as e:
                stat.warning(f"⚠️ SMOTE skipped: {e}")
        pb.progress(20)

        # ── Class weights ────────────────────────────────────────────
        cw = None
        if use_class_weight:
            classes = np.unique(y_tr)
            weights = compute_class_weight('balanced', classes=classes, y=y_tr)
            cw = {int(c): float(w) for c, w in zip(classes, weights)}

        # ── Build model ──────────────────────────────────────────────
        stat.info("🏗 Building ANN…")
        model = keras.Sequential()
        model.add(kl.Input(shape=(X_tr.shape[1],)))
        for i, sz in enumerate(layer_sizes):
            model.add(kl.Dense(sz, activation=activation,
                               kernel_regularizer=regularizers.l2(1e-3), name=f"h{i+1}"))
            if dropout > 0:
                model.add(kl.Dropout(dropout, name=f"do{i+1}"))
            model.add(kl.BatchNormalization(name=f"bn{i+1}"))
        model.add(kl.Dense(1, activation='sigmoid', name='out'))

        opt_map = {
            'adam':    keras.optimizers.Adam(lr),
            'sgd':     keras.optimizers.SGD(lr, momentum=0.9),
            'rmsprop': keras.optimizers.RMSprop(lr),
            'adamax':  keras.optimizers.Adamax(lr),
        }
        model.compile(optimizer=opt_map[optimizer],
                      loss='binary_crossentropy',
                      metrics=['accuracy',
                               keras.metrics.AUC(name='auc'),
                               keras.metrics.Precision(name='precision'),
                               keras.metrics.Recall(name='recall')])
        pb.progress(28)

        # ── Live training ────────────────────────────────────────────
        stat.info("🚀 Training… watch the curves update live!")
        hist_d = {k: [] for k in ['loss','val_loss','accuracy','val_accuracy',
                                   'auc','val_auc','precision','recall']}
        chart_ph = st.empty()
        log_ph   = st.empty()

        class LC(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                for k in hist_d:
                    if k in logs:
                        hist_d[k].append(logs[k])
                pb.progress(int(28 + (epoch+1)/epochs * 62))
                ep_x = list(range(1, len(hist_d['loss'])+1))
                if ep_x:
                    fl = make_subplots(1, 2, subplot_titles=["Loss", "Acc & AUC"])
                    fl.add_trace(go.Scatter(x=ep_x, y=hist_d['loss'], name='Train Loss',
                        line=dict(color='#ff3366', width=2)), 1, 1)
                    if hist_d['val_loss']:
                        fl.add_trace(go.Scatter(x=ep_x, y=hist_d['val_loss'], name='Val Loss',
                            line=dict(color='#ff8099', width=1.5, dash='dash')), 1, 1)
                    fl.add_trace(go.Scatter(x=ep_x, y=hist_d['accuracy'], name='Train Acc',
                        line=dict(color='#00ff88', width=2)), 1, 2)
                    if hist_d['val_accuracy']:
                        fl.add_trace(go.Scatter(x=ep_x, y=hist_d['val_accuracy'], name='Val Acc',
                            line=dict(color='#88ffcc', width=1.5, dash='dash')), 1, 2)
                    if hist_d['auc']:
                        fl.add_trace(go.Scatter(x=ep_x, y=hist_d['auc'], name='AUC',
                            line=dict(color='#7c3aff', width=2)), 1, 2)
                    for r, c in [(1,1),(1,2)]:
                        fl.update_xaxes(gridcolor='#2a2a40', row=r, col=c)
                        fl.update_yaxes(gridcolor='#2a2a40', row=r, col=c)
                    fl.update_layout(paper_bgcolor='#1a1a26', plot_bgcolor='#1a1a26',
                        font_color='#e8e8f0', height=270,
                        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8e8f0', size=9)),
                        margin=dict(t=30, b=20, l=40, r=20))
                    chart_ph.plotly_chart(fl, use_container_width=True)
                log_ph.markdown(
                    f"<div class='tlog'>Epoch {epoch+1:03d}/{epochs}  │  "
                    f"loss:{logs.get('loss',0):.4f}  val_loss:{logs.get('val_loss',0):.4f}  │  "
                    f"acc:{logs.get('accuracy',0):.4f}  auc:{logs.get('auc',0):.4f}  "
                    f"precision:{logs.get('precision',0):.4f}  recall:{logs.get('recall',0):.4f}</div>",
                    unsafe_allow_html=True)

        model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                  validation_split=0.15, class_weight=cw,
                  callbacks=[LC(), keras.callbacks.EarlyStopping(
                      patience=12, restore_best_weights=True, monitor='val_auc', mode='max')],
                  verbose=0)
        pb.progress(95)

        # ── Evaluate with tuned threshold ────────────────────────────
        y_prob = model.predict(X_te, verbose=0).flatten()
        y_hat  = (y_prob >= threshold).astype(int)   # ← tuned threshold
        y_hat_default = (y_prob >= 0.5).astype(int)  # for comparison

        report         = classification_report(y_te, y_hat, output_dict=True, zero_division=0)
        report_default = classification_report(y_te, y_hat_default, output_dict=True, zero_division=0)
        auc_  = roc_auc_score(y_te, y_prob)
        ap_   = average_precision_score(y_te, y_prob)
        cm_   = confusion_matrix(y_te, y_hat)

        st.session_state.update(dict(
            model=model, scaler=sc, le_map=le_map, feat_cols=feat_cols,
            X_te=X_te, y_te=y_te, y_prob=y_prob, y_hat=y_hat,
            report=report, report_default=report_default,
            auc=auc_, ap=ap_, cm=cm_, hist=hist_d,
            smote=smote_ok, threshold=threshold,
            class_weight=cw
        ))
        pb.progress(100)
        stat.success(
            f"✅ Done!  ROC-AUC: {auc_:.4f}  │  Avg Precision: {ap_:.4f}  │  "
            f"Fraud Recall @ thresh={threshold}: {report.get('1',{}).get('recall',0):.4f}")
        st.info("👉 Go to **Results & Metrics** tab for full evaluation.")

    else:
        st.markdown('<div class="wbox">⚡ Set hyperparameters in the sidebar and click '
                    '<b>🚀 TRAIN MODEL</b>.<br><br>'
                    '📌 With 0.15% fraud rate, <b>SMOTE + Class Weights + low threshold</b> '
                    'are critical — all enabled by default.</div>', unsafe_allow_html=True)

        st.markdown("<div class='st-title'>Imbalance Strategy Explained</div>", unsafe_allow_html=True)
        for title, desc in [
            ("1 · 50,000 Rows",    "Full dataset loaded — only 76 fraud cases out of 50,000 (0.15%)"),
            ("2 · SMOTE",          "Creates synthetic fraud samples to balance training data"),
            ("3 · Class Weights",  "Penalizes misclassifying fraud more heavily during training"),
            ("4 · Low Threshold",  "Classify as fraud if score ≥ 0.3 instead of 0.5 — catches more fraud"),
            ("5 · Focus Metric",   "Use ROC-AUC & Fraud-F1 — accuracy is misleading at 0.15% fraud rate"),
        ]:
            st.markdown(f"""<div style="background:#1a1a26;border:1px solid #2a2a40;
              border-left:3px solid #7c3aff;border-radius:8px;padding:12px 16px;margin-bottom:8px;">
              <div style="font-weight:700;color:#e8e8f0;font-size:0.87rem;">{title}</div>
              <div style="color:#888899;font-size:0.76rem;margin-top:4px;
                          font-family:'Space Mono',monospace;">{desc}</div></div>""",
              unsafe_allow_html=True)

# ══ TAB 3 — RESULTS ══════════════════════════════════════════════════════════
with tab3:
    if 'model' not in st.session_state:
        st.markdown('<div class="wbox" style="text-align:center;padding:48px;margin-top:32px;">'
                    '🧠 No model trained yet.<br><br>Go to <b>Train ANN</b> tab.</div>',
                    unsafe_allow_html=True)
    else:
        r      = st.session_state['report']
        r_def  = st.session_state['report_default']
        auc_   = st.session_state['auc']
        ap_    = st.session_state['ap']
        y_prob = st.session_state['y_prob']
        y_te   = st.session_state['y_te']
        y_hat  = st.session_state['y_hat']
        cm_    = st.session_state['cm']
        thr    = st.session_state['threshold']

        # Threshold comparison banner
        rec_tuned   = r.get('1', {}).get('recall', 0)
        rec_default = r_def.get('1', {}).get('recall', 0)
        f1_tuned    = r.get('1', {}).get('f1-score', 0)
        f1_default  = r_def.get('1', {}).get('f1-score', 0)

        st.markdown(f"""<div class="ibox">
        🎯 <b>Threshold = {thr} vs default 0.5 comparison</b><br>
        Fraud Recall:  default 0.5 → <b>{rec_default:.4f}</b>  │  tuned {thr} → <b style="color:#00ff88">{rec_tuned:.4f}</b><br>
        Fraud F1:      default 0.5 → <b>{f1_default:.4f}</b>   │  tuned {thr} → <b style="color:#00ff88">{f1_tuned:.4f}</b>
        </div>""", unsafe_allow_html=True)

        # KPI row
        for col, (lab, val, clr) in zip(st.columns(6), [
            ("ROC AUC",      f"{auc_:.4f}",                                    "#7c3aff"),
            ("Avg Precision",f"{ap_:.4f}",                                     "#00ff88"),
            ("F1 Fraud",     f"{r.get('1',{}).get('f1-score',0):.4f}",         "#ff3366"),
            ("Recall Fraud", f"{r.get('1',{}).get('recall',0):.4f}",           "#ffaa00"),
            ("Precision Fr.",f"{r.get('1',{}).get('precision',0):.4f}",        "#00ff88"),
            ("Threshold",    f"{thr}",                                          "#7c3aff"),
        ]):
            with col:
                st.markdown(f'<div class="mc"><div class="mv" style="color:{clr};">{val}</div>'
                            f'<div class="ml">{lab}</div></div>', unsafe_allow_html=True)

        st.markdown("<div class='st-title'>Evaluation Plots</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fig_cm = go.Figure(go.Heatmap(
                z=cm_, x=['Pred:Legit','Pred:Fraud'], y=['Act:Legit','Act:Fraud'],
                colorscale=[[0,'#1a1a26'],[0.5,'#7c3aff'],[1,'#ff3366']],
                showscale=False, text=cm_,
                texttemplate="<b>%{text}</b>",
                textfont=dict(size=22, color='white', family='Space Mono')))
            fig_cm.update_layout(**CL, height=290,
                title=dict(text=f"Confusion Matrix (threshold={thr})",
                           font=dict(color='#888899', size=13)),
                yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig_cm, use_container_width=True)

        with c2:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds_roc = roc_curve(y_te, y_prob)
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy',
                fillcolor='rgba(124,58,255,0.1)',
                line=dict(color='#7c3aff', width=2.5), name=f'AUC={auc_:.3f}'))
            fig_r.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                line=dict(color='#444', dash='dash', width=1), showlegend=False))
            fig_r.update_layout(**CL, height=290,
                title=dict(text="ROC Curve", font=dict(color='#888899', size=13)),
                xaxis=dict(title='FPR', gridcolor='#2a2a40'),
                yaxis=dict(title='TPR', gridcolor='#2a2a40'))
            st.plotly_chart(fig_r, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            from sklearn.metrics import precision_recall_curve
            prec, rec, thr_pr = precision_recall_curve(y_te, y_prob)
            fig_pr = go.Figure(go.Scatter(x=rec, y=prec, mode='lines', fill='tozeroy',
                fillcolor='rgba(0,255,136,0.07)',
                line=dict(color='#00ff88', width=2.5), name=f'AP={ap_:.4f}'))
            # Mark current threshold on PR curve
            fig_pr.update_layout(**CL, height=290,
                title=dict(text="Precision-Recall Curve (crucial for imbalanced data)",
                           font=dict(color='#888899', size=12)),
                xaxis=dict(title='Recall', gridcolor='#2a2a40'),
                yaxis=dict(title='Precision', gridcolor='#2a2a40'))
            st.plotly_chart(fig_pr, use_container_width=True)

        with c4:
            # Score distribution — zoomed to show fraud scores clearly
            fig_d = go.Figure()
            for flag, nm, clr in [(0,'Legit','rgba(0,255,136,0.5)'),
                                   (1,'Fraud','rgba(255,51,102,0.85)')]:
                fig_d.add_trace(go.Histogram(x=y_prob[y_te==flag], name=nm,
                    nbinsx=50, marker_color=clr, opacity=0.85))
            fig_d.add_vline(x=thr, line_dash='dash', line_color='#ffaa00',
                annotation_text=f'Threshold={thr}',
                annotation_font=dict(color='#ffaa00', size=9, family='Space Mono'))
            fig_d.add_vline(x=0.5, line_dash='dot', line_color='#888899',
                annotation_text='Default=0.5',
                annotation_font=dict(color='#888899', size=9, family='Space Mono'))
            fig_d.update_layout(**CL, barmode='overlay', height=290,
                title=dict(text="Prediction Score Distribution", font=dict(color='#888899', size=13)),
                xaxis=dict(title='Predicted Probability', gridcolor='#2a2a40'),
                yaxis=dict(title='Count', gridcolor='#2a2a40'))
            st.plotly_chart(fig_d, use_container_width=True)

        # Threshold sensitivity analysis
        st.markdown("<div class='st-title'>Threshold Sensitivity Analysis</div>", unsafe_allow_html=True)
        thresholds_range = np.arange(0.05, 0.95, 0.05)
        recalls, precisions, f1s, fprs_list = [], [], [], []
        for t in thresholds_range:
            yh = (y_prob >= t).astype(int)
            rep = classification_report(y_te, yh, output_dict=True, zero_division=0)
            recalls.append(rep.get('1',{}).get('recall', 0))
            precisions.append(rep.get('1',{}).get('precision', 0))
            f1s.append(rep.get('1',{}).get('f1-score', 0))

        fig_thr = go.Figure()
        fig_thr.add_trace(go.Scatter(x=thresholds_range, y=recalls, name='Recall (Fraud)',
            line=dict(color='#ff3366', width=2.5)))
        fig_thr.add_trace(go.Scatter(x=thresholds_range, y=precisions, name='Precision (Fraud)',
            line=dict(color='#00ff88', width=2.5)))
        fig_thr.add_trace(go.Scatter(x=thresholds_range, y=f1s, name='F1 (Fraud)',
            line=dict(color='#7c3aff', width=2.5)))
        fig_thr.add_vline(x=thr, line_dash='dash', line_color='#ffaa00',
            annotation_text=f'Current: {thr}',
            annotation_font=dict(color='#ffaa00', size=10, family='Space Mono'))
        fig_thr.update_layout(**CL, height=320,
            title=dict(text="How Threshold Affects Fraud Detection Performance",
                       font=dict(color='#888899', size=13)),
            xaxis=dict(title='Decision Threshold', gridcolor='#2a2a40'),
            yaxis=dict(title='Score', gridcolor='#2a2a40', range=[0,1]))
        st.plotly_chart(fig_thr, use_container_width=True)

        # Classification report tables — tuned vs default
        st.markdown("<div class='st-title'>Classification Report</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        for col, rep, title in [
            (col_a, r,     f"Tuned Threshold = {thr}"),
            (col_b, r_def, "Default Threshold = 0.5")]:
            with col:
                st.markdown(f"**{title}**")
                cr = pd.DataFrame({
                    'Class': ['Legit (0)','Fraud (1)','Macro Avg','Weighted Avg'],
                    'Precision': [rep.get('0',{}).get('precision',0), rep.get('1',{}).get('precision',0),
                        rep.get('macro avg',{}).get('precision',0), rep.get('weighted avg',{}).get('precision',0)],
                    'Recall': [rep.get('0',{}).get('recall',0), rep.get('1',{}).get('recall',0),
                        rep.get('macro avg',{}).get('recall',0), rep.get('weighted avg',{}).get('recall',0)],
                    'F1-Score': [rep.get('0',{}).get('f1-score',0), rep.get('1',{}).get('f1-score',0),
                        rep.get('macro avg',{}).get('f1-score',0), rep.get('weighted avg',{}).get('f1-score',0)],
                    'Support': [int(rep.get('0',{}).get('support',0)), int(rep.get('1',{}).get('support',0)),
                        int(rep.get('macro avg',{}).get('support',0)), int(rep.get('weighted avg',{}).get('support',0))],
                })
                cr[['Precision','Recall','F1-Score']] = cr[['Precision','Recall','F1-Score']].applymap(lambda x: f"{x:.4f}")
                st.dataframe(cr.set_index('Class'), use_container_width=True)

        # Training history
        h = st.session_state['hist']
        if h['loss']:
            st.markdown("<div class='st-title'>Full Training History</div>", unsafe_allow_html=True)
            ep_x = list(range(1, len(h['loss'])+1))
            fh2  = make_subplots(1, 2, subplot_titles=['Loss','Accuracy & AUC'])
            fh2.add_trace(go.Scatter(x=ep_x, y=h['loss'], name='Train Loss',
                line=dict(color='#ff3366', width=2)), 1, 1)
            if h['val_loss']:
                fh2.add_trace(go.Scatter(x=ep_x, y=h['val_loss'], name='Val Loss',
                    line=dict(color='#ff8099', width=1.5, dash='dash')), 1, 1)
            fh2.add_trace(go.Scatter(x=ep_x, y=h['accuracy'], name='Acc',
                line=dict(color='#00ff88', width=2)), 1, 2)
            if h['val_accuracy']:
                fh2.add_trace(go.Scatter(x=ep_x, y=h['val_accuracy'], name='Val Acc',
                    line=dict(color='#88ffcc', width=1.5, dash='dash')), 1, 2)
            if h['auc']:
                fh2.add_trace(go.Scatter(x=ep_x, y=h['auc'], name='AUC',
                    line=dict(color='#7c3aff', width=2)), 1, 2)
            for r2, c2 in [(1,1),(1,2)]:
                fh2.update_xaxes(gridcolor='#2a2a40', row=r2, col=c2)
                fh2.update_yaxes(gridcolor='#2a2a40', row=r2, col=c2)
            fh2.update_layout(paper_bgcolor='#1a1a26', plot_bgcolor='#1a1a26',
                font_color='#e8e8f0', height=310,
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#e8e8f0', size=9)),
                margin=dict(t=35, b=20, l=40, r=20))
            st.plotly_chart(fh2, use_container_width=True)

# ══ TAB 4 — PREDICTOR ════════════════════════════════════════════════════════
with tab4:
    if 'model' not in st.session_state:
        st.markdown('<div class="wbox" style="text-align:center;padding:48px;margin-top:32px;">'
                    '🧠 Train a model first.</div>', unsafe_allow_html=True)
    else:
        thr = st.session_state['threshold']
        st.markdown("<div class='st-title'>Live Fraud Predictor</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class="ibox">
        Current model uses threshold = <b>{thr}</b>.
        Any transaction scoring ≥ {thr} will be flagged as fraud.
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            txn_type = st.selectbox("Transaction Type",  df['transaction_type'].unique())
            merchant = st.selectbox("Merchant Category", df['merchant_category'].unique())
            amount   = st.number_input("Amount (INR)", 1, 500000, 5000)
        with c2:
            s_age  = st.selectbox("Sender Age Group",   df['sender_age_group'].unique())
            r_age  = st.selectbox("Receiver Age Group", df['receiver_age_group'].unique())
            device = st.selectbox("Device Type",         df['device_type'].unique())
        with c3:
            hour = st.slider("Hour of Day", 0, 23, 14)
            isw  = st.selectbox("Is Weekend?", [0,1], format_func=lambda x: "Yes" if x else "No")

        if st.button("🔍 PREDICT FRAUD RISK", use_container_width=True):
            try:
                le_map = st.session_state['le_map']
                def se(col, val):
                    try:    return le_map[col].transform([str(val)])[0]
                    except: return 0

                inp = np.array([[se('transaction_type', txn_type),
                                 se('merchant_category', merchant),
                                 amount,
                                 se('sender_age_group', s_age),
                                 se('receiver_age_group', r_age),
                                 hour, isw,
                                 se('device_type', device)]])
                inp_sc = st.session_state['scaler'].transform(inp)
                prob   = float(st.session_state['model'].predict(inp_sc, verbose=0)[0][0])
                pct    = prob * 100
                flagged = prob >= thr

                clr     = '#ff3366' if flagged else '#00ff88'
                verdict = ('🚨 FRAUD FLAGGED' if flagged else '✅ LEGITIMATE')
                risk_label = ('HIGH' if prob >= 0.5 else ('MEDIUM' if prob >= thr else 'LOW'))

                st.markdown(f"""<div style="background:#1a1a26;border:2px solid {clr};border-radius:16px;
                  padding:36px;text-align:center;margin:20px 0;">
                  <div style="font-size:3.5rem;font-weight:800;color:{clr};
                              font-family:'Space Mono',monospace;">{pct:.2f}%</div>
                  <div style="font-size:0.72rem;color:#888899;letter-spacing:3px;
                              text-transform:uppercase;margin:8px 0;">Fraud Probability Score</div>
                  <div style="font-size:1.3rem;color:{clr};font-weight:800;margin-top:12px;">{verdict}</div>
                  <div style="margin-top:18px;background:#0a0a0f;border-radius:8px;padding:12px;
                              font-family:'Space Mono',monospace;font-size:0.72rem;color:#888899;line-height:1.8;">
                    raw score: {prob:.6f}  │  threshold: {thr}  │  risk level: {risk_label}
                  </div></div>""", unsafe_allow_html=True)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=pct,
                    domain={'x':[0,1],'y':[0,1]},
                    title={'text':"Fraud Risk Score",
                           'font':{'color':'#888899','family':'Syne','size':14}},
                    number={'font':{'color':clr,'family':'Space Mono'},'suffix':'%'},
                    gauge={'axis':{'range':[0,100],'tickcolor':'#888899'},
                           'bar':{'color':clr},
                           'bgcolor':'#1a1a26','bordercolor':'#2a2a40',
                           'steps':[
                               {'range':[0,   thr*100], 'color':'rgba(0,255,136,0.07)'},
                               {'range':[thr*100, 50],  'color':'rgba(255,170,0,0.07)'},
                               {'range':[50,  100],     'color':'rgba(255,51,102,0.12)'}],
                           'threshold':{'line':{'color':'#ffaa00','width':3},
                                        'thickness':0.75,'value':thr*100}}))
                fig_g.update_layout(paper_bgcolor='#1a1a26', font_color='#e8e8f0',
                    height=270, margin=dict(t=40,b=20,l=40,r=40))
                st.plotly_chart(fig_g, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")