import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Robo-Advisor | FGV 2026",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 4px solid #3b82f6;
}
.main-header h1 {
    color: #f8fafc;
    font-size: 1.8rem;
    margin: 0 0 .3rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    margin: 0;
    font-size: .9rem;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 1rem;
}
.metric-card {
    flex: 1;
    background: #1e293b;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #334155;
    text-align: center;
}
.metric-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    margin: .2rem 0;
}
.metric-card .lbl {
    font-size: .75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: .5px;
}
.metric-card .sub {
    font-size: .72rem;
    color: #475569;
}
.green { color: #4ade80; }
.red   { color: #f87171; }
.blue  { color: #60a5fa; }
.amber { color: #fbbf24; }

/* Section titles */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #3b82f6;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: .4rem;
    margin: 1.2rem 0 .8rem 0;
}

/* Alert box */
.alert-box {
    background: #0f2a1f;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: .75rem 1rem;
    color: #4ade80;
    font-size: .85rem;
    margin-top: .75rem;
}
.alert-warn {
    background: #1c1208;
    border-color: #92400e;
    color: #fbbf24;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f172a !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: .82rem;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: .72rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-left: 6px;
}
.badge-green { background: #14532d; color: #4ade80; }
.badge-blue  { background: #1e3a5f; color: #60a5fa; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
TICKERS = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BOVA11.SA", "SPY", "GLD", "BND"]
NOMES   = {
    "PETR4.SA": "Petrobras", "VALE3.SA": "Vale", "ITUB4.SA": "Itau",
    "WEGE3.SA": "WEG", "BOVA11.SA": "BOVA11", "SPY": "SP500", "GLD": "Ouro", "BND": "RendaFixa"
}
CDI = 0.1075
CAPITAL_DEFAULT = 100_000

# ── Funções utilitárias ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def carregar_dados(start="2018-01-01", end="2024-12-31"):
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    precos = raw["Close"].dropna(how="all").ffill().dropna()
    precos.columns = [NOMES.get(t, t) for t in precos.columns]
    return precos

def portfolio_perf(pesos, med, cov, rf=CDI):
    p = np.array(pesos)
    ret = float(np.dot(p, med))
    vol = float(np.sqrt(np.dot(p.T, np.dot(cov, p))))
    sr  = (ret - rf) / vol if vol > 0 else 0
    return ret, vol, sr

def neg_sharpe(pesos, med, cov, rf=CDI):
    return -portfolio_perf(pesos, med, cov, rf)[2]

def otimizar(med, cov, rf=CDI, max_peso=0.4):
    n = len(med)
    w0 = np.ones(n) / n
    bounds = tuple((0, max_peso) for _ in range(n))
    cons   = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    res = minimize(neg_sharpe, w0, args=(med, cov, rf),
                   method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else w0

def calcular_drawdown(ret):
    cum = (1 + ret).cumprod()
    return cum / cum.cummax() - 1

def metricas_portfolio(ret, rf=CDI):
    ra  = float((1 + ret).prod() - 1)
    raa = float((1 + ra) ** (252 / len(ret)) - 1)
    vol = float(ret.std() * np.sqrt(252))
    sr  = (raa - rf) / vol if vol > 0 else 0
    mdd = float(calcular_drawdown(ret).min())
    cal = raa / abs(mdd) if mdd != 0 else 0
    return {"Ret Acum (%)": round(ra*100,2), "Ret Anual (%)": round(raa*100,2),
            "Vol (%)": round(vol*100,2), "Sharpe": round(sr,3),
            "MDD (%)": round(mdd*100,2), "Calmar": round(cal,3)}

def var_cvar(ret_arr, conf=0.95, capital=CAPITAL_DEFAULT):
    v   = float(np.percentile(ret_arr, (1-conf)*100))
    cv  = float(ret_arr[ret_arr <= v].mean())
    return {"VaR 95% (%)": round(v*100,3), "CVaR 95% (%)": round(cv*100,3),
            "VaR R$": round(abs(v)*capital,2), "CVaR R$": round(abs(cv)*capital,2)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parâmetros")
    st.markdown("---")

    perfil = st.radio("Perfil do Investidor",
                      ["Conservador", "Moderado", "Arrojado", "Agressivo"], index=1)

    perfil_cfg = {
        "Conservador": (2,  10, 60, 5),
        "Moderado":    (5,  20, 20, 10),
        "Arrojado":    (7,  10, 10, 10),
        "Agressivo":   (9,  5,   5,  5),
    }
    risco_def, rf_def, int_def, alt_def = perfil_cfg[perfil]

    capital    = st.slider("Capital Inicial (R$ mil)", 10, 2000, 100, step=10) * 1_000
    horizonte  = st.slider("Horizonte (anos)", 1, 20, 5)
    risco_tol  = st.slider("Tolerância a Risco (1–10)", 1, 10, risco_def)
    pct_rf     = st.slider("% Renda Fixa", 0, 80, rf_def, step=5)
    pct_int    = st.slider("% Internacional", 0, 50, int_def, step=5)
    pct_alt    = st.slider("% Alternativos", 0, 30, alt_def, step=5)
    max_peso   = st.slider("Peso Máximo por Ativo (%)", 10, 60, 40, step=5) / 100

    st.markdown("---")
    st.markdown("**Período dos dados**")
    data_inicio = st.selectbox("Início", ["2015-01-01","2018-01-01","2020-01-01"], index=1)
    st.markdown("---")
    st.caption("FGV · IA Aplicada ao Mercado Financeiro · 2026")

# ── Carregar dados ────────────────────────────────────────────────────────────
with st.spinner("Carregando dados históricos via Yahoo Finance..."):
    precos   = carregar_dados(start=data_inicio)
    retornos = precos.pct_change().dropna()
    N        = len(retornos.columns)
    med_ret  = retornos.mean() * 252
    cov_mat  = retornos.cov()  * 252
    ret_an   = med_ret
    vol_an   = retornos.std() * np.sqrt(252)

# ── Otimização ────────────────────────────────────────────────────────────────
pesos_otimos = otimizar(med_ret, cov_mat, CDI, max_peso)
ret_p, vol_p, sr_p = portfolio_perf(pesos_otimos, med_ret, cov_mat)
var_p = -vol_p / np.sqrt(252) * 1.645

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
  <h1>📈 Robo-Advisor
    <span class="badge badge-green">FGV 2026</span>
    <span class="badge badge-blue">IA + Markowitz</span>
  </h1>
  <p>Otimização de Portfólio com Inteligência Artificial · Perfil: <strong style="color:#60a5fa">{perfil}</strong> · {len(precos)} dias de dados · {len(precos.columns)} ativos</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Portfolio Ótimo",
    "📊 Análise de Dados",
    "🤖 Machine Learning",
    "⏱️ Backtesting",
    "🔴 Risco & Stress",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO ÓTIMO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Métricas principais
    sharpe_color = "green" if sr_p > 0 else "red"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="lbl">Retorno Esperado</div>
        <div class="val green">{ret_p*100:.1f}%</div>
        <div class="sub">ao ano</div>
      </div>
      <div class="metric-card">
        <div class="lbl">Volatilidade</div>
        <div class="val blue">{vol_p*100:.1f}%</div>
        <div class="sub">anualizada</div>
      </div>
      <div class="metric-card">
        <div class="lbl">Sharpe Ratio</div>
        <div class="val {sharpe_color}">{sr_p:.3f}</div>
        <div class="sub">vs CDI 10,75%</div>
      </div>
      <div class="metric-card">
        <div class="lbl">VaR Diário 95%</div>
        <div class="val red">{var_p*100:.2f}%</div>
        <div class="sub">paramétrico</div>
      </div>
      <div class="metric-card">
        <div class="lbl">Ativos</div>
        <div class="val amber">{N}</div>
        <div class="sub">diversificados</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Alocação Recomendada pela IA</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ativos_ord = sorted(zip(retornos.columns, pesos_otimos), key=lambda x: x[1])
        ativos_n, pesos_n = zip(*[(a, p) for a, p in ativos_ord if p > 0.003])
        cores = plt.cm.Blues(np.linspace(0.4, 0.95, len(ativos_n)))
        bars = ax.barh(ativos_n, [p*100 for p in pesos_n], color=cores, height=0.6)
        for bar, val in zip(bars, pesos_n):
            ax.text(val*100 + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va="center", color="white", fontsize=9)
        ax.set_xlabel("Peso (%)", color="#64748b", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.xaxis.label.set_color("#64748b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Projeção Patrimonial</div>', unsafe_allow_html=True)
        anos_p = np.arange(0, horizonte + 1)
        r_bova = float(ret_an.get("BOVA11", ret_an.iloc[4]))
        pat_robo = capital * (1 + ret_p) ** anos_p
        pat_bova = capital * (1 + r_bova) ** anos_p

        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.plot(anos_p, pat_robo/1000, color="#3b82f6", linewidth=2.5, label="Robo-Advisor", marker="o", markersize=4)
        ax.plot(anos_p, pat_bova/1000, color="#64748b", linewidth=1.5, linestyle="--", label="Ibovespa", marker="s", markersize=3)
        ax.fill_between(anos_p, pat_bova/1000, pat_robo/1000, alpha=0.15, color="#3b82f6")
        ax.set_xlabel("Anos", color="#64748b", fontsize=9)
        ax.set_ylabel("Patrimônio (R$ mil)", color="#64748b", fontsize=9)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        alpha_final = (pat_robo[-1] - pat_bova[-1]) / 1000
        st.markdown(f"""
        <div class="alert-box">
          Em <strong>{horizonte} anos</strong>: Robo-Advisor → <strong>R$ {pat_robo[-1]/1000:.0f}k</strong>
          vs Ibovespa → R$ {pat_bova[-1]/1000:.0f}k · Alpha acumulado: <strong>+R$ {alpha_final:.0f}k</strong>
        </div>
        """, unsafe_allow_html=True)

    # Fronteira Eficiente
    st.markdown('<div class="section-title">Fronteira Eficiente (Monte Carlo — 8.000 portfólios)</div>', unsafe_allow_html=True)
    N_MC = 8000
    np.random.seed(42)
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(N_MC):
        w = np.random.dirichlet(np.ones(N))
        r_, v_, s_ = portfolio_perf(w, med_ret, cov_mat)
        mc_r.append(r_); mc_v.append(v_); mc_s.append(s_)
    mc_r = np.array(mc_r); mc_v = np.array(mc_v); mc_s = np.array(mc_s)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    sc = ax.scatter(mc_v*100, mc_r*100, c=mc_s, cmap="YlOrRd", alpha=0.35, s=6)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Ratio", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8", fontsize=8)

    ax.scatter(vol_p*100, ret_p*100, color="#3b82f6", s=300, marker="*", zorder=10,
               label=f"Máx Sharpe ({sr_p:.2f})", edgecolors="white", linewidth=0.8)
    for i, ativo in enumerate(retornos.columns):
        ax.scatter(vol_an[ativo]*100, ret_an[ativo]*100, s=60, color="#f59e0b", alpha=0.8, zorder=8)
        ax.annotate(ativo, (vol_an[ativo]*100, ret_an[ativo]*100),
                    textcoords="offset points", xytext=(5,3), fontsize=8, color="#94a3b8")

    x_cml = np.linspace(0, mc_v.max()*100, 100)
    ax.plot(x_cml, CDI*100 + sr_p*x_cml, "b--", alpha=0.5, linewidth=1.2, label="Capital Market Line")

    ax.set_xlabel("Volatilidade (%)", color="#64748b", fontsize=10)
    ax.set_ylabel("Retorno Anualizado (%)", color="#64748b", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
    for sp in ax.spines.values(): sp.set_color("#1e293b")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISE DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Evolução dos Preços (Base 100)</div>', unsafe_allow_html=True)
    precos_norm = precos / precos.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    cores_linha = plt.cm.tab10(np.linspace(0, 1, len(precos_norm.columns)))
    for i, col in enumerate(precos_norm.columns):
        ax.plot(precos_norm.index, precos_norm[col], label=col, linewidth=1.4, color=cores_linha[i])
    ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8", ncol=4)
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values(): sp.set_color("#1e293b")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Matriz de Correlação</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        sns.heatmap(retornos.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                    center=0, ax=ax, vmin=-1, vmax=1, square=True,
                    annot_kws={"size":8}, linewidths=.3)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Risco vs Retorno Anualizado</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        for i, col in enumerate(ret_an.index):
            ax.scatter(vol_an[col]*100, ret_an[col]*100, s=110, zorder=5, color=cores_linha[i])
            ax.annotate(col, (vol_an[col]*100, ret_an[col]*100),
                        textcoords="offset points", xytext=(6,3), fontsize=9, color="#94a3b8")
        ax.axhline(CDI*100, color="#64748b", linestyle=":", alpha=0.6, linewidth=1)
        ax.set_xlabel("Volatilidade (%)", color="#64748b", fontsize=9)
        ax.set_ylabel("Retorno (%)", color="#64748b", fontsize=9)
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">Estatísticas dos Ativos</div>', unsafe_allow_html=True)
    stats_df = pd.DataFrame({
        "Retorno Anual (%)":  (ret_an*100).round(2),
        "Volatilidade (%)":   (vol_an*100).round(2),
        "Sharpe (rf=0)":      (ret_an/vol_an).round(3),
        "Peso Ótimo (%)":     (pd.Series(pesos_otimos, index=retornos.columns)*100).round(1),
    }).sort_values("Sharpe (rf=0)", ascending=False)
    st.dataframe(stats_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Predição de Retorno Esperado com ML</div>', unsafe_allow_html=True)
    st.info("Os modelos usam features de momentum, volatilidade e RSI para prever o retorno médio dos próximos 21 dias.")

    @st.cache_data(show_spinner=False)
    def treinar_modelos(retornos_hash):
        def criar_features(ret_ativo):
            df = pd.DataFrame({"ret": ret_ativo})
            df["ret_1d"]   = df["ret"].shift(1)
            df["ret_5d"]   = df["ret"].rolling(5).mean().shift(1)
            df["ret_21d"]  = df["ret"].rolling(21).mean().shift(1)
            df["vol_21d"]  = df["ret"].rolling(21).std().shift(1)
            df["vol_63d"]  = df["ret"].rolling(63).std().shift(1)
            df["mom_63"]   = df["ret"].rolling(63).sum().shift(1)
            delta = df["ret"].shift(1)
            g = delta.clip(lower=0).rolling(14).mean()
            p = (-delta.clip(upper=0)).rolling(14).mean()
            df["rsi_14"]  = 100 - (100 / (1 + g / (p + 1e-10)))
            df["skew_21"] = df["ret"].rolling(21).skew().shift(1)
            df["target"]  = df["ret"].rolling(21).mean().shift(-21)
            return df.dropna()

        frames = []
        for col in retornos.columns:
            d = criar_features(retornos[col])
            d["ativo"] = col
            frames.append(d)
        df_ml = pd.concat(frames).dropna()
        FEAT = ["ret_1d","ret_5d","ret_21d","vol_21d","vol_63d","mom_63","rsi_14","skew_21"]
        X = df_ml[FEAT].values; y = df_ml["target"].values
        split = int(len(X)*0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
        modelos = {
            "Regressão Linear":  LinearRegression(),
            "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
        }
        res = {}
        for nome, mod in modelos.items():
            mod.fit(X_tr_s, y_tr)
            pred = mod.predict(X_te_s)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            r2   = r2_score(y_te, pred)
            acc  = np.mean(np.sign(pred) == np.sign(y_te))
            res[nome] = {"RMSE": rmse, "R2": r2, "Acc": acc, "model": mod, "feat": FEAT}
        return res

    with st.spinner("Treinando modelos de ML..."):
        resultados_ml = treinar_modelos(str(retornos.shape))

    # Tabela de resultados
    df_res = pd.DataFrame([
        {"Modelo": k, "RMSE": round(v["RMSE"],5), "R²": round(v["R2"],4),
         "Acurácia Direcional (%)": round(v["Acc"]*100,1)}
        for k, v in resultados_ml.items()
    ]).set_index("Modelo")
    st.dataframe(df_res, use_container_width=True)

    melhor = max(resultados_ml, key=lambda x: resultados_ml[x]["Acc"])
    acc_m  = resultados_ml[melhor]["Acc"]*100
    cor_alerta = "alert-box" if acc_m > 52 else "alert-warn"
    st.markdown(f'<div class="{cor_alerta}">Melhor modelo: <strong>{melhor}</strong> — Acurácia Direcional: <strong>{acc_m:.1f}%</strong> (baseline aleatório: 50%)</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Comparação de Métricas</div>', unsafe_allow_html=True)
        nomes_m = list(resultados_ml.keys())
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), facecolor="#0f172a")
        for ax in axes: ax.set_facecolor("#0f172a")
        cores_b = ["#3b82f6","#ef4444","#22c55e"]
        axes[0].bar(range(3), [resultados_ml[m]["RMSE"] for m in nomes_m], color=cores_b)
        axes[0].set_title("RMSE", color="#94a3b8", fontsize=9)
        axes[1].bar(range(3), [resultados_ml[m]["R2"] for m in nomes_m], color=cores_b)
        axes[1].set_title("R²", color="#94a3b8", fontsize=9)
        axes[2].bar(range(3), [resultados_ml[m]["Acc"]*100 for m in nomes_m], color=cores_b)
        axes[2].axhline(50, color="white", linestyle="--", alpha=0.5, linewidth=1)
        axes[2].set_title("Acurácia Dir. (%)", color="#94a3b8", fontsize=9)
        for ax in axes:
            ax.set_xticks(range(3))
            ax.set_xticklabels(["Lin.","RF","GB"], color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8")
            for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-title">Importância das Features (RF)</div>', unsafe_allow_html=True)
        rf_mod = resultados_ml["Random Forest"]["model"]
        feat   = resultados_ml["Random Forest"]["feat"]
        imp = pd.Series(rf_mod.feature_importances_, index=feat).sort_values()
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        cores_imp = plt.cm.Blues(np.linspace(0.4, 0.95, len(imp)))
        imp.plot(kind="barh", ax=ax, color=cores_imp)
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Backtesting com Rebalanceamento Mensal</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def executar_backtest(retornos_hash, pesos_hash):
        pesos_at = np.array(pesos_otimos)
        ret_port = []
        for i in range(len(retornos)):
            if i % 21 == 0 and i > 252:
                ret_j = retornos.iloc[max(0,i-252):i]
                try:
                    opt = minimize(neg_sharpe, pesos_at,
                                   args=(ret_j.mean()*252, ret_j.cov()*252, CDI),
                                   method="SLSQP",
                                   bounds=tuple((0, max_peso) for _ in range(N)),
                                   constraints=[{"type":"eq","fun":lambda x: np.sum(x)-1}])
                    if opt.success: pesos_at = opt.x
                except: pass
            ret_port.append(float(np.dot(retornos.iloc[i].values, pesos_at)))
        return pd.Series(ret_port, index=retornos.index, name="Robo-Advisor")

    with st.spinner("Executando backtesting (pode demorar ~1 min)..."):
        ret_robo  = executar_backtest(str(retornos.shape), str(pesos_otimos.sum()))
        ret_equal = (retornos * (1/N)).sum(axis=1).rename("Equal Weight")
        col_bova  = [c for c in retornos.columns if "bov" in c.lower() or "BOVA" in c.upper()]
        ret_bova  = retornos[col_bova[0] if col_bova else retornos.columns[4]].rename("Ibovespa")

    cum_robo  = (1 + ret_robo).cumprod() * 100
    cum_equal = (1 + ret_equal).cumprod() * 100
    cum_bova  = (1 + ret_bova).cumprod() * 100
    dd_robo   = calcular_drawdown(ret_robo) * 100
    dd_bova   = calcular_drawdown(ret_bova) * 100

    # Tabela de métricas
    df_met = pd.DataFrame([
        metricas_portfolio(ret_robo),
        metricas_portfolio(ret_equal),
        metricas_portfolio(ret_bova),
    ], index=["Robo-Advisor","Equal Weight","Ibovespa"])
    st.dataframe(df_met, use_container_width=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), facecolor="#0f172a")
    for ax in axes: ax.set_facecolor("#0f172a")

    axes[0].plot(cum_robo.index,  cum_robo,  color="#3b82f6", linewidth=2, label="Robo-Advisor")
    axes[0].plot(cum_equal.index, cum_equal, color="#f59e0b", linewidth=1.5, linestyle="--", label="Equal Weight")
    axes[0].plot(cum_bova.index,  cum_bova,  color="#64748b", linewidth=1.5, linestyle=":", label="Ibovespa")
    axes[0].set_ylabel("Patrimônio (Base 100)", color="#64748b", fontsize=9)
    axes[0].set_title("Curva de Patrimônio", color="#94a3b8", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")

    axes[1].fill_between(dd_robo.index, dd_robo, 0, alpha=0.5, color="#3b82f6", label="Robo-Advisor")
    axes[1].fill_between(dd_bova.index, dd_bova, 0, alpha=0.3, color="#64748b", label="Ibovespa")
    axes[1].set_ylabel("Drawdown (%)", color="#64748b", fontsize=9)
    axes[1].set_title("Drawdown Histórico", color="#94a3b8", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")

    ret_m = ret_robo.resample("ME").apply(lambda x: (1+x).prod()-1) * 100
    cores_m = ["#ef4444" if v < 0 else "#22c55e" for v in ret_m]
    axes[2].bar(ret_m.index, ret_m.values, color=cores_m, alpha=0.8, width=20)
    axes[2].axhline(0, color="white", linewidth=0.7)
    axes[2].set_ylabel("Retorno Mensal (%)", color="#64748b", fontsize=9)
    axes[2].set_title("Retornos Mensais — Robo-Advisor", color="#94a3b8", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.tick_params(colors="#94a3b8", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e293b")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISCO & STRESS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Value at Risk & CVaR (95% confiança)</div>', unsafe_allow_html=True)

    cap_risk = capital
    vc_robo = var_cvar(ret_robo.values, capital=cap_risk)
    vc_bova = var_cvar(ret_bova.values, capital=cap_risk)

    df_var = pd.DataFrame([vc_robo, vc_bova], index=["Robo-Advisor","Ibovespa"])
    st.dataframe(df_var, use_container_width=True)
    st.caption(f"Valores em R$ calculados sobre capital de R$ {cap_risk:,.0f}")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.hist(ret_robo.values*100, bins=70, color="#3b82f6", alpha=0.6, density=True)
        ax.axvline(vc_robo["VaR 95% (%)"],  color="#ef4444", linewidth=2,
                   label=f'VaR 95%: {vc_robo["VaR 95% (%)"]:.2f}%')
        ax.axvline(vc_robo["CVaR 95% (%)"], color="#b91c1c", linewidth=2, linestyle="--",
                   label=f'CVaR 95%: {vc_robo["CVaR 95% (%)"]:.2f}%')
        ax.set_title("Distribuição Retornos — Robo-Advisor", color="#94a3b8", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
        ax.tick_params(colors="#94a3b8"); 
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        ax.hist(ret_bova.values*100, bins=70, color="#64748b", alpha=0.6, density=True)
        ax.axvline(vc_bova["VaR 95% (%)"],  color="#ef4444", linewidth=2,
                   label=f'VaR 95%: {vc_bova["VaR 95% (%)"]:.2f}%')
        ax.axvline(vc_bova["CVaR 95% (%)"], color="#b91c1c", linewidth=2, linestyle="--",
                   label=f'CVaR 95%: {vc_bova["CVaR 95% (%)"]:.2f}%')
        ax.set_title("Distribuição Retornos — Ibovespa", color="#94a3b8", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">Stress Test — Crises Históricas</div>', unsafe_allow_html=True)
    cenarios = {
        "COVID Mar-Abr 2020":  ("2020-02-01","2020-04-30"),
        "Crise BR 2015":       ("2015-05-01","2015-10-31"),
        "Alta Juros EUA 2022": ("2022-01-01","2022-10-31"),
        "Crise Turca 2018":    ("2018-08-01","2018-11-30"),
    }
    rows = []
    for nome, (ini, fim) in cenarios.items():
        mask = (ret_robo.index >= ini) & (ret_robo.index <= fim)
        if mask.sum() < 5: continue
        rp = float((1+ret_robo[mask]).prod()-1)
        rb = float((1+ret_bova[mask]).prod()-1)
        mp = float(calcular_drawdown(ret_robo[mask]).min())
        mb = float(calcular_drawdown(ret_bova[mask]).min())
        rows.append({"Cenário": nome,
                     "Robo-Advisor (%)": round(rp*100,1), "Ibovespa (%)": round(rb*100,1),
                     "MDD Robo (%)": round(mp*100,1),     "MDD Ibov (%)": round(mb*100,1)})
    if rows:
        df_stress = pd.DataFrame(rows).set_index("Cenário")
        st.dataframe(df_stress, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        x_st = np.arange(len(df_stress))
        w = 0.35
        ax.bar(x_st - w/2, df_stress["Robo-Advisor (%)"], w, color="#3b82f6", alpha=0.85, label="Robo-Advisor")
        ax.bar(x_st + w/2, df_stress["Ibovespa (%)"],     w, color="#64748b", alpha=0.85, label="Ibovespa")
        ax.axhline(0, color="white", linewidth=0.7)
        ax.set_xticks(x_st)
        ax.set_xticklabels(df_stress.index, color="#94a3b8", fontsize=9, rotation=10)
        ax.set_ylabel("Retorno no Período (%)", color="#64748b", fontsize=9)
        ax.set_title("Performance em Crises Históricas", color="#94a3b8", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_color("#1e293b")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Simulação Monte Carlo de cenário de crise
    st.markdown('<div class="section-title">Simulação Monte Carlo — Cenário de Crise (-30% mercado)</div>', unsafe_allow_html=True)
    np.random.seed(99)
    N_SIM = 5000
    dias_crise = 60
    ret_medio_crise = -0.002
    vol_crise = vol_p * 1.8 / np.sqrt(252)
    simulacoes = np.random.normal(ret_medio_crise, vol_crise, (N_SIM, dias_crise))
    pat_final = capital * (1 + simulacoes).prod(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.hist(pat_final/1000, bins=80, color="#3b82f6", alpha=0.6, density=True)
    ax.axvline(capital/1000, color="white", linewidth=1.5, linestyle="--", label="Capital inicial")
    p5 = np.percentile(pat_final/1000, 5)
    ax.axvline(p5, color="#ef4444", linewidth=2, label=f"Percentil 5%: R$ {p5:.0f}k")
    ax.set_xlabel("Patrimônio Final (R$ mil)", color="#64748b", fontsize=9)
    ax.set_ylabel("Densidade", color="#64748b", fontsize=9)
    ax.set_title(f"Monte Carlo — {N_SIM} cenários de crise (60 dias)", color="#94a3b8", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values(): sp.set_color("#1e293b")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    prob_perda = (pat_final < capital).mean() * 100
    perda_med  = (capital - pat_final[pat_final < capital]).mean() / 1000 if (pat_final < capital).any() else 0
    st.markdown(f"""
    <div class="alert-warn">
      Em cenário de crise simulado: probabilidade de perda = <strong>{prob_perda:.1f}%</strong> ·
      Perda média nos cenários negativos = <strong>R$ {perda_med:.1f}k</strong>
    </div>
    """, unsafe_allow_html=True)
