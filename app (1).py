import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3 { font-family: 'IBM Plex Mono', monospace !important; }
.main-header { background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%); padding:1.5rem 2rem;
  border-radius:10px; margin-bottom:1rem; border-left:4px solid #3b82f6; }
.main-header h1 { color:#f8fafc; font-size:1.6rem; margin:0 0 .2rem 0; }
.main-header p  { color:#94a3b8; margin:0; font-size:.85rem; }
.metric-row { display:flex; gap:10px; margin-bottom:1rem; }
.metric-card { flex:1; background:#1e293b; border-radius:8px; padding:.8rem 1rem;
  border:1px solid #334155; text-align:center; }
.metric-card .val { font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:600; margin:.15rem 0; }
.metric-card .lbl { font-size:.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.5px; }
.metric-card .sub { font-size:.68rem; color:#475569; }
.green{color:#4ade80} .red{color:#f87171} .blue{color:#60a5fa} .amber{color:#fbbf24}
.section-title { font-family:'IBM Plex Mono',monospace; font-size:.75rem; text-transform:uppercase;
  letter-spacing:1.5px; color:#3b82f6; border-bottom:1px solid #1e3a5f; padding-bottom:.3rem; margin:1rem 0 .6rem 0; }
.alert-box { background:#0f2a1f; border:1px solid #166534; border-radius:6px;
  padding:.6rem .9rem; color:#4ade80; font-size:.82rem; margin-top:.6rem; }
.alert-warn { background:#1c1208; border-color:#92400e; color:#fbbf24; }
[data-testid="stSidebar"] { background:#0f172a !important; }
</style>
""", unsafe_allow_html=True)

# ── Constantes ────────────────────────────────────────────────────────────────
TICKERS = ['PETR4.SA','VALE3.SA','ITUB4.SA','WEGE3.SA','BOVA11.SA','SPY','GLD','BND']
NOMES   = {'PETR4.SA':'Petrobras','VALE3.SA':'Vale','ITUB4.SA':'Itau',
           'WEGE3.SA':'WEG','BOVA11.SA':'BOVA11','SPY':'SP500','GLD':'Ouro','BND':'RendaFixa'}
NOMES_REV = {v:k for k,v in NOMES.items()}
CDI = 0.1075

# ── Parâmetros anuais realistas por ativo (fallback) ─────────────────────────
PARAMS = {
    'Petrobras': (0.18, 0.42), 'Vale':     (0.12, 0.33),
    'Itau':      (0.14, 0.24), 'WEG':      (0.22, 0.29),
    'BOVA11':    (0.10, 0.22), 'SP500':    (0.14, 0.16),
    'Ouro':      (0.09, 0.15), 'RendaFixa':(0.04, 0.07),
}
CORR_BASE = np.array([
    [1.00, 0.65, 0.58, 0.45, 0.72, 0.28, 0.12, -0.05],
    [0.65, 1.00, 0.52, 0.40, 0.68, 0.30, 0.18, -0.04],
    [0.58, 0.52, 1.00, 0.48, 0.70, 0.25, 0.08, -0.02],
    [0.45, 0.40, 0.48, 1.00, 0.55, 0.38, 0.10, -0.01],
    [0.72, 0.68, 0.70, 0.55, 1.00, 0.32, 0.15, -0.03],
    [0.28, 0.30, 0.25, 0.38, 0.32, 1.00, 0.22,  0.10],
    [0.12, 0.18, 0.08, 0.10, 0.15, 0.22, 1.00,  0.05],
    [-0.05,-0.04,-0.02,-0.01,-0.03, 0.10, 0.05,  1.00],
])

# ── Funções de dados ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def carregar_dados(start='2018-01-01', end='2024-12-31'):
    """Tenta yfinance; se falhar gera dados simulados realistas."""
    try:
        import yfinance as yf
        raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)

        # Lidar com MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            precos = raw['Close'].copy()
        else:
            precos = raw.copy()

        # Renomear colunas para nomes amigáveis
        precos = precos.rename(columns=NOMES)

        # Manter só colunas que existem em PARAMS
        cols_validas = [c for c in precos.columns if c in PARAMS]
        precos = precos[cols_validas]

        # Remover colunas completamente vazias
        precos = precos.dropna(axis=1, how='all')
        precos = precos.ffill().dropna()

        if len(precos) < 100 or len(precos.columns) < 3:
            raise ValueError("Dados insuficientes")

        return precos, False  # False = dados reais

    except Exception:
        return _gerar_simulado(start, end), True  # True = dados simulados

def _gerar_simulado(start='2018-01-01', end='2024-12-31'):
    """Gera série histórica simulada com parâmetros realistas."""
    np.random.seed(42)
    datas = pd.bdate_range(start=start, end=end)
    n = len(datas)
    ativos = list(PARAMS.keys())
    n_at = len(ativos)

    # Matriz de covariância diária
    vols_d = np.array([PARAMS[a][1] / np.sqrt(252) for a in ativos])
    cov_d  = CORR_BASE * np.outer(vols_d, vols_d)

    # Garantir positiva definida
    eigvals, eigvecs = np.linalg.eigh(cov_d)
    eigvals = np.maximum(eigvals, 1e-8)
    cov_d = eigvecs @ np.diag(eigvals) @ eigvecs.T

    L = np.linalg.cholesky(cov_d)
    Z = np.random.randn(n, n_at)
    ret_d = Z @ L.T + np.array([PARAMS[a][0] / 252 for a in ativos])

    precos = pd.DataFrame(100 * np.cumprod(1 + ret_d, axis=0),
                          index=datas, columns=ativos)
    return precos

# ── Funções de otimização ─────────────────────────────────────────────────────
def portfolio_perf(pesos, med, cov, rf=CDI):
    p   = np.array(pesos)
    ret = float(np.dot(p, med))
    vol = float(np.sqrt(max(float(p @ cov @ p), 1e-10)))
    sr  = (ret - rf) / vol
    return ret, vol, sr

def neg_sharpe(pesos, med, cov, rf=CDI):
    return -portfolio_perf(pesos, med, cov, rf)[2]

def otimizar(med, cov, rf=CDI, max_w=0.40):
    n   = len(med)
    w0  = np.ones(n) / n
    bds = tuple((0.0, max_w) for _ in range(n))
    con = [{'type':'eq','fun': lambda x: np.sum(x)-1}]
    res = minimize(neg_sharpe, w0, args=(med, cov, rf),
                   method='SLSQP', bounds=bds, constraints=con,
                   options={'ftol':1e-9,'maxiter':1000})
    return res.x if res.success else w0

def calc_dd(ret):
    cum = (1 + ret).cumprod()
    return cum / cum.cummax() - 1

def metricas(ret, rf=CDI):
    if len(ret) < 2:
        return {k: 0.0 for k in ['Ret Acum (%)','Ret Anual (%)','Vol (%)','Sharpe','MDD (%)','Calmar']}
    ra  = float((1 + ret).prod() - 1)
    raa = float((1 + ra) ** (252 / max(len(ret), 1)) - 1)
    vol = float(ret.std() * np.sqrt(252))
    sr  = (raa - rf) / vol if vol > 0 else 0.0
    mdd = float(calc_dd(ret).min())
    cal = raa / abs(mdd) if mdd != 0 else 0.0
    return {'Ret Acum (%)':round(ra*100,2),'Ret Anual (%)':round(raa*100,2),
            'Vol (%)':round(vol*100,2),'Sharpe':round(sr,3),
            'MDD (%)':round(mdd*100,2),'Calmar':round(cal,3)}

def var_cvar(arr, conf=0.95, capital=100_000):
    if len(arr) < 10:
        return {'VaR 95% (%)':0,'CVaR 95% (%)':0,'VaR R$':0,'CVaR R$':0}
    v  = float(np.percentile(arr, (1-conf)*100))
    cv = float(arr[arr <= v].mean()) if (arr <= v).any() else v
    return {'VaR 95% (%)':round(v*100,3),'CVaR 95% (%)':round(cv*100,3),
            'VaR R$':round(abs(v)*capital,2),'CVaR R$':round(abs(cv)*capital,2)}

# ── Estilo dos gráficos ───────────────────────────────────────────────────────
def fig_dark(w=12, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor='#0f172a')
    ax.set_facecolor('#0f172a')
    return fig, ax

def style_ax(ax):
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.xaxis.label.set_color('#64748b')
    ax.yaxis.label.set_color('#64748b')
    for sp in ax.spines.values():
        sp.set_color('#1e293b')

CORES = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#84cc16']

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Parâmetros")
    st.markdown("---")

    perfil = st.radio("Perfil do Investidor",
                      ["Conservador","Moderado","Arrojado","Agressivo"], index=1)
    cfg = {"Conservador":(2,55,10,5),"Moderado":(5,25,20,10),
           "Arrojado":(7,10,15,10),"Agressivo":(9,5,10,5)}
    rdef,rfdef,intdef,altdef = cfg[perfil]

    capital   = st.slider("Capital Inicial (R$ mil)", 10, 2000, 100, step=10) * 1_000
    horizonte = st.slider("Horizonte (anos)", 1, 20, 5)
    risco_tol = st.slider("Tolerância a Risco (1–10)", 1, 10, rdef)
    pct_rf    = st.slider("% Renda Fixa", 0, 80, rfdef, step=5)
    pct_int   = st.slider("% Internacional", 0, 50, intdef, step=5)
    pct_alt   = st.slider("% Alternativos (Ouro)", 0, 30, altdef, step=5)
    max_w     = st.slider("Peso Máximo por Ativo (%)", 10, 60, 40, step=5) / 100
    st.markdown("---")
    st.caption("FGV · IA Aplicada ao Mercado Financeiro · 2026")

# ══════════════════════════════════════════════════════════════════════════════
# CARREGAR DADOS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Carregando dados..."):
    precos, simulado = carregar_dados()

if simulado:
    st.info("ℹ️ Usando dados simulados com parâmetros históricos reais (yfinance indisponível no momento).")

# Garantir que precos é DataFrame limpo
precos = precos.copy()
if not isinstance(precos.index, pd.DatetimeIndex):
    precos.index = pd.to_datetime(precos.index)

retornos = precos.pct_change().dropna()

# Verificação de segurança
assert len(retornos) > 50, "Dados insuficientes"
assert len(retornos.columns) >= 3, "Poucos ativos"

ativos   = list(retornos.columns)
N        = len(ativos)
med_ret  = retornos.mean() * 252
cov_mat  = retornos.cov()  * 252
ret_an   = med_ret.copy()
vol_an   = retornos.std()  * np.sqrt(252)

# ── Otimização ────────────────────────────────────────────────────────────────
pesos_ot = otimizar(med_ret.values, cov_mat.values, CDI, max_w)
ret_p, vol_p, sr_p = portfolio_perf(pesos_ot, med_ret.values, cov_mat.values)
var_d = -vol_p / np.sqrt(252) * 1.645

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="main-header">
  <h1>📈 Robo-Advisor
    <span style="background:#14532d;color:#4ade80;padding:2px 10px;border-radius:12px;font-size:.7rem;margin-left:8px">FGV 2026</span>
    <span style="background:#1e3a5f;color:#60a5fa;padding:2px 10px;border-radius:12px;font-size:.7rem;margin-left:4px">IA + Markowitz</span>
  </h1>
  <p>Otimização de Portfólio com Inteligência Artificial · Perfil: <strong style="color:#60a5fa">{perfil}</strong>
     · {len(precos)} dias de dados · {N} ativos</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Portfólio Ótimo","📊 Análise de Dados",
    "🤖 Machine Learning","⏱️ Backtesting","🔴 Risco & Stress"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFÓLIO ÓTIMO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    sc = "green" if sr_p > 0 else "red"
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="lbl">Retorno Esperado</div>
        <div class="val green">{ret_p*100:.1f}%</div><div class="sub">ao ano</div></div>
      <div class="metric-card"><div class="lbl">Volatilidade</div>
        <div class="val blue">{vol_p*100:.1f}%</div><div class="sub">anualizada</div></div>
      <div class="metric-card"><div class="lbl">Sharpe Ratio</div>
        <div class="val {sc}">{sr_p:.3f}</div><div class="sub">vs CDI 10,75%</div></div>
      <div class="metric-card"><div class="lbl">VaR Diário 95%</div>
        <div class="val red">{var_d*100:.2f}%</div><div class="sub">paramétrico</div></div>
      <div class="metric-card"><div class="lbl">Ativos</div>
        <div class="val amber">{N}</div><div class="sub">diversificados</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Alocação Recomendada pela IA</div>', unsafe_allow_html=True)
        pesos_df = pd.Series(pesos_ot, index=ativos).sort_values()
        pesos_df = pesos_df[pesos_df > 0.005]
        fig, ax = fig_dark(6, 4)
        bars = ax.barh(pesos_df.index, pesos_df.values * 100,
                       color=CORES[:len(pesos_df)], height=0.6)
        for bar, val in zip(bars, pesos_df.values):
            ax.text(val*100 + 0.3, bar.get_y()+bar.get_height()/2,
                    f"{val*100:.1f}%", va='center', color='white', fontsize=9)
        ax.set_xlabel("Peso (%)")
        style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-title">Projeção Patrimonial</div>', unsafe_allow_html=True)

        # Calcular retorno do benchmark (BOVA11 ou primeiro ativo BR)
        bova_cols = [c for c in ativos if 'bova' in c.lower() or 'bov' in c.lower()]
        if bova_cols:
            r_bench = float(ret_an[bova_cols[0]])
        else:
            r_bench = float(ret_an.iloc[4]) if len(ret_an) > 4 else 0.08

        anos_p = np.arange(0, horizonte + 1)
        pat_r = capital * (1 + ret_p) ** anos_p
        pat_b = capital * (1 + r_bench) ** anos_p

        fig, ax = fig_dark(6, 4)
        ax.plot(anos_p, pat_r/1000, color='#3b82f6', lw=2.5, marker='o', ms=4, label='Robo-Advisor')
        ax.plot(anos_p, pat_b/1000, color='#64748b', lw=1.5, ls='--', marker='s', ms=3, label='Benchmark')
        ax.fill_between(anos_p, pat_b/1000, pat_r/1000, alpha=0.15, color='#3b82f6')
        ax.set_xlabel("Anos"); ax.set_ylabel("Patrimônio (R$ mil)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('R$%.0fk'))
        ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
        style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        alpha_k = (pat_r[-1] - pat_b[-1]) / 1000
        st.markdown(f"""<div class="alert-box">
          Em <strong>{horizonte} anos</strong>: Robo-Advisor → <strong>R$ {pat_r[-1]/1000:.0f}k</strong>
          vs Benchmark → R$ {pat_b[-1]/1000:.0f}k · Alpha: <strong>+R$ {alpha_k:.0f}k</strong>
        </div>""", unsafe_allow_html=True)

    # Fronteira Eficiente
    st.markdown('<div class="section-title">Fronteira Eficiente — Monte Carlo (8.000 portfólios)</div>', unsafe_allow_html=True)
    np.random.seed(42)
    mc_r, mc_v, mc_s = [], [], []
    for _ in range(8000):
        w = np.random.dirichlet(np.ones(N))
        r_, v_, s_ = portfolio_perf(w, med_ret.values, cov_mat.values)
        mc_r.append(r_); mc_v.append(v_); mc_s.append(s_)
    mc_r = np.array(mc_r); mc_v = np.array(mc_v); mc_s = np.array(mc_s)

    fig, ax = fig_dark(11, 5)
    sc_plot = ax.scatter(mc_v*100, mc_r*100, c=mc_s, cmap='YlOrRd', alpha=0.35, s=6)
    cb = plt.colorbar(sc_plot, ax=ax)
    cb.set_label('Sharpe Ratio', color='#94a3b8', fontsize=9)
    cb.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#94a3b8', fontsize=8)
    ax.scatter(vol_p*100, ret_p*100, color='#3b82f6', s=300, marker='*', zorder=10,
               label=f'Portfólio Ótimo (Sharpe={sr_p:.2f})', edgecolors='white', lw=0.8)
    for i, ativo in enumerate(ativos):
        ax.scatter(vol_an[ativo]*100, ret_an[ativo]*100, s=60, color='#f59e0b', alpha=0.8, zorder=8)
        ax.annotate(ativo, (vol_an[ativo]*100, ret_an[ativo]*100),
                    textcoords='offset points', xytext=(5,3), fontsize=8, color='#94a3b8')
    x_cml = np.linspace(0, mc_v.max()*100, 100)
    ax.plot(x_cml, CDI*100 + sr_p*x_cml, 'b--', alpha=0.5, lw=1.2, label='Capital Market Line')
    ax.set_xlabel("Volatilidade Anualizada (%)"); ax.set_ylabel("Retorno Anualizado (%)")
    ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
    style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANÁLISE DE DADOS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Evolução dos Preços (Base 100)</div>', unsafe_allow_html=True)

    # CORREÇÃO: normalizar corretamente
    primeiro_valido = precos.apply(lambda col: col.dropna().iloc[0] if col.dropna().shape[0] > 0 else 1.0)
    precos_norm = precos.divide(primeiro_valido, axis=1) * 100

    fig, ax = fig_dark(12, 4)
    for i, col in enumerate(precos_norm.columns):
        serie = precos_norm[col].dropna()
        ax.plot(serie.index, serie.values, label=col, lw=1.4, color=CORES[i % len(CORES)])
    ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8', ncol=4)
    ax.set_ylabel("Preço normalizado (base 100)")
    style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Matriz de Correlação</div>', unsafe_allow_html=True)
        fig, ax = fig_dark(6, 5)
        corr = retornos.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, ax=ax, vmin=-1, vmax=1, square=True,
                    annot_kws={'size':8}, linewidths=.3)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-title">Risco vs Retorno Anualizado</div>', unsafe_allow_html=True)
        fig, ax = fig_dark(6, 5)
        for i, col in enumerate(ativos):
            ax.scatter(float(vol_an[col])*100, float(ret_an[col])*100,
                       s=110, zorder=5, color=CORES[i % len(CORES)])
            ax.annotate(col, (float(vol_an[col])*100, float(ret_an[col])*100),
                        textcoords='offset points', xytext=(6,3), fontsize=9, color='#94a3b8')
        ax.axhline(CDI*100, color='#64748b', ls=':', alpha=0.6, lw=1)
        ax.set_xlabel("Volatilidade Anualizada (%)"); ax.set_ylabel("Retorno Anualizado (%)")
        style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown('<div class="section-title">Estatísticas dos Ativos</div>', unsafe_allow_html=True)
    stats_df = pd.DataFrame({
        'Retorno Anual (%)': (ret_an*100).round(2),
        'Volatilidade (%)':  (vol_an*100).round(2),
        'Sharpe (rf=CDI)':   ((ret_an - CDI) / vol_an).round(3),
        'Peso Ótimo (%)':    (pd.Series(pesos_ot, index=ativos)*100).round(1),
    }).sort_values('Sharpe (rf=CDI)', ascending=False)
    st.dataframe(stats_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Predição de Retorno com Machine Learning</div>', unsafe_allow_html=True)
    st.info("Os modelos usam indicadores técnicos para prever o sinal do retorno futuro de cada ativo.")

    @st.cache_data(show_spinner=False)
    def treinar_ml(cache_key):
        def feats(ret_s):
            df = pd.DataFrame({'ret': ret_s})
            df['ret_1d']  = df['ret'].shift(1)
            df['ret_5d']  = df['ret'].rolling(5).mean().shift(1)
            df['ret_21d'] = df['ret'].rolling(21).mean().shift(1)
            df['vol_21d'] = df['ret'].rolling(21).std().shift(1)
            df['vol_63d'] = df['ret'].rolling(63).std().shift(1)
            df['mom_63']  = df['ret'].rolling(63).sum().shift(1)
            g = df['ret'].shift(1).clip(lower=0).rolling(14).mean()
            p = (-df['ret'].shift(1).clip(upper=0)).rolling(14).mean()
            df['rsi_14'] = 100 - (100/(1 + g/(p+1e-10)))
            df['skew_21'] = df['ret'].rolling(21).skew().shift(1)
            df['target']  = df['ret'].rolling(21).mean().shift(-21)
            return df.dropna()

        frames = []
        for col in retornos.columns:
            d = feats(retornos[col].copy())
            d['ativo'] = col
            frames.append(d)
        df_ml = pd.concat(frames).dropna()

        FEAT = ['ret_1d','ret_5d','ret_21d','vol_21d','vol_63d','mom_63','rsi_14','skew_21']
        X = df_ml[FEAT].values; y = df_ml['target'].values
        split = int(len(X)*0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        mods = {
            'Regressão Linear':  LinearRegression(),
            'Random Forest':     RandomForestRegressor(n_estimators=150, max_depth=6,
                                                        min_samples_leaf=20, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                            learning_rate=0.05, random_state=42),
        }
        res = {}
        for nome, mod in mods.items():
            mod.fit(X_tr_s, y_tr)
            pred = mod.predict(X_te_s)
            res[nome] = {
                'RMSE': round(np.sqrt(mean_squared_error(y_te, pred)), 5),
                'R2':   round(r2_score(y_te, pred), 4),
                'Acc':  round(float(np.mean(np.sign(pred) == np.sign(y_te)))*100, 1),
                'model': mod, 'feat': FEAT
            }
        return res

    with st.spinner("Treinando modelos de ML..."):
        res_ml = treinar_ml(str(retornos.shape))

    df_res = pd.DataFrame([
        {'Modelo':k,'RMSE':v['RMSE'],'R²':v['R2'],'Acurácia Direcional (%)':v['Acc']}
        for k,v in res_ml.items()
    ]).set_index('Modelo')
    st.dataframe(df_res, use_container_width=True)

    melhor = max(res_ml, key=lambda x: res_ml[x]['Acc'])
    acc_m  = res_ml[melhor]['Acc']
    cls = "alert-box" if acc_m > 52 else "alert-warn"
    st.markdown(f'<div class="{cls}">Melhor modelo: <strong>{melhor}</strong> — Acurácia Direcional: <strong>{acc_m:.1f}%</strong> (baseline aleatório = 50%)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Comparação de Métricas</div>', unsafe_allow_html=True)
        nomes_m = list(res_ml.keys())
        fig, axes = plt.subplots(1,3, figsize=(9,3.5), facecolor='#0f172a')
        for ax in axes: ax.set_facecolor('#0f172a')
        cb3 = ['#3b82f6','#ef4444','#22c55e']
        axes[0].bar(range(3),[res_ml[m]['RMSE'] for m in nomes_m], color=cb3)
        axes[0].set_title('RMSE', color='#94a3b8', fontsize=9)
        axes[1].bar(range(3),[res_ml[m]['R2'] for m in nomes_m], color=cb3)
        axes[1].set_title('R²', color='#94a3b8', fontsize=9)
        axes[2].bar(range(3),[res_ml[m]['Acc'] for m in nomes_m], color=cb3)
        axes[2].axhline(50, color='white', ls='--', alpha=0.5, lw=1)
        axes[2].set_title('Acurácia Dir. (%)', color='#94a3b8', fontsize=9)
        for ax in axes:
            ax.set_xticks(range(3))
            ax.set_xticklabels(['Lin.','RF','GB'], color='#94a3b8', fontsize=8)
            ax.tick_params(colors='#94a3b8')
            for sp in ax.spines.values(): sp.set_color('#1e293b')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown('<div class="section-title">Importância das Features (RF)</div>', unsafe_allow_html=True)
        rf_m = res_ml['Random Forest']['model']
        feat = res_ml['Random Forest']['feat']
        imp  = pd.Series(rf_m.feature_importances_, index=feat).sort_values()
        fig, ax = fig_dark(6, 3.5)
        imp.plot(kind='barh', ax=ax, color=plt.cm.Blues(np.linspace(0.4,0.95,len(imp))))
        style_ax(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTESTING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Backtesting com Rebalanceamento Mensal</div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def executar_bt(cache_key):
        pw = pesos_ot.copy()
        ret_p_list = []
        for i in range(len(retornos)):
            if i % 21 == 0 and i > 252:
                rj = retornos.iloc[max(0,i-252):i]
                try:
                    pw2 = otimizar(rj.mean().values*252, rj.cov().values*252, CDI, max_w)
                    pw = pw2
                except Exception:
                    pass
            ret_p_list.append(float(np.dot(retornos.iloc[i].values, pw)))
        return pd.Series(ret_p_list, index=retornos.index, name='Robo-Advisor')

    with st.spinner("Executando backtesting (~1 min)..."):
        ret_robo  = executar_bt(str(retornos.shape) + str(pesos_ot.sum()))
        ret_equal = (retornos * (1/N)).sum(axis=1).rename('Equal Weight')
        bova_c    = [c for c in ativos if 'bov' in c.lower()]
        ret_bova  = retornos[bova_c[0] if bova_c else ativos[min(4,N-1)]].rename('Benchmark')

    df_met = pd.DataFrame([
        metricas(ret_robo), metricas(ret_equal), metricas(ret_bova)
    ], index=['Robo-Advisor','Equal Weight','Benchmark'])
    st.dataframe(df_met, use_container_width=True)

    cum_r = (1+ret_robo).cumprod()*100
    cum_e = (1+ret_equal).cumprod()*100
    cum_b = (1+ret_bova).cumprod()*100
    dd_r  = calc_dd(ret_robo)*100
    dd_b  = calc_dd(ret_bova)*100

    fig, axes = plt.subplots(3,1, figsize=(12,12), facecolor='#0f172a')
    for ax in axes: ax.set_facecolor('#0f172a')

    axes[0].plot(cum_r.index, cum_r, color='#3b82f6', lw=2, label='Robo-Advisor')
    axes[0].plot(cum_e.index, cum_e, color='#f59e0b', lw=1.5, ls='--', label='Equal Weight')
    axes[0].plot(cum_b.index, cum_b, color='#64748b', lw=1.5, ls=':', label='Benchmark')
    axes[0].set_ylabel("Patrimônio (Base 100)"); axes[0].legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
    axes[0].set_title("Curva de Patrimônio", color='#94a3b8', fontsize=11, fontweight='bold')

    axes[1].fill_between(dd_r.index, dd_r, 0, alpha=0.5, color='#3b82f6', label='Robo-Advisor')
    axes[1].fill_between(dd_b.index, dd_b, 0, alpha=0.3, color='#64748b', label='Benchmark')
    axes[1].set_ylabel("Drawdown (%)"); axes[1].legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
    axes[1].set_title("Drawdown Histórico", color='#94a3b8', fontsize=11, fontweight='bold')

    ret_m = ret_robo.resample('ME').apply(lambda x: (1+x).prod()-1)*100
    axes[2].bar(ret_m.index, ret_m.values,
                color=['#ef4444' if v<0 else '#22c55e' for v in ret_m], alpha=0.8, width=20)
    axes[2].axhline(0, color='white', lw=0.7)
    axes[2].set_ylabel("Retorno Mensal (%)")
    axes[2].set_title("Retornos Mensais — Robo-Advisor", color='#94a3b8', fontsize=11, fontweight='bold')

    for ax in axes:
        style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISCO & STRESS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Value at Risk & CVaR (95% confiança)</div>', unsafe_allow_html=True)

    vc_r = var_cvar(ret_robo.values,  capital=capital)
    vc_b = var_cvar(ret_bova.values,  capital=capital)
    df_v = pd.DataFrame([vc_r,vc_b], index=['Robo-Advisor','Benchmark'])
    st.dataframe(df_v, use_container_width=True)
    st.caption(f"Valores em R$ sobre capital de R$ {capital:,.0f}")

    c1, c2 = st.columns(2)
    for col, (ret_s, nome, vc) in zip([c1,c2],[
        (ret_robo, 'Robo-Advisor', vc_r),
        (ret_bova, 'Benchmark',    vc_b)
    ]):
        with col:
            fig, ax = fig_dark(6, 4)
            ax.hist(ret_s.values*100, bins=70, color='#3b82f6' if nome=='Robo-Advisor' else '#64748b',
                    alpha=0.6, density=True)
            ax.axvline(vc['VaR 95% (%)'],  color='#ef4444', lw=2,
                       label=f'VaR 95%: {vc["VaR 95% (%)"]:.2f}%')
            ax.axvline(vc['CVaR 95% (%)'], color='#b91c1c', lw=2, ls='--',
                       label=f'CVaR 95%: {vc["CVaR 95% (%)"]:.2f}%')
            ax.set_title(f'Distribuição — {nome}', color='#94a3b8', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
            style_ax(ax)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown('<div class="section-title">Stress Test — Crises Históricas</div>', unsafe_allow_html=True)
    cenarios = {
        'COVID Mar-Abr 2020':   ('2020-02-01','2020-04-30'),
        'Crise BR 2015':        ('2015-05-01','2015-10-31'),
        'Alta Juros EUA 2022':  ('2022-01-01','2022-10-31'),
        'Crise 2018':           ('2018-08-01','2018-11-30'),
    }
    rows_st = []
    for nome, (ini,fim) in cenarios.items():
        mask = (ret_robo.index >= ini) & (ret_robo.index <= fim)
        if mask.sum() < 5: continue
        rp = float((1+ret_robo[mask]).prod()-1)
        rb = float((1+ret_bova[mask]).prod()-1)
        mp = float(calc_dd(ret_robo[mask]).min())
        mb = float(calc_dd(ret_bova[mask]).min())
        rows_st.append({'Cenário':nome,'Robo-Advisor (%)':round(rp*100,1),'Benchmark (%)':round(rb*100,1),
                        'MDD Robo (%)':round(mp*100,1),'MDD Benchmark (%)':round(mb*100,1)})
    if rows_st:
        st.dataframe(pd.DataFrame(rows_st).set_index('Cenário'), use_container_width=True)

    st.markdown('<div class="section-title">Monte Carlo — Cenário de Crise (60 dias)</div>', unsafe_allow_html=True)
    np.random.seed(99)
    N_SIM   = 5000
    vol_c   = vol_p * 1.8 / np.sqrt(252)
    sims    = np.random.normal(-0.002, vol_c, (N_SIM, 60))
    pat_fin = capital * (1 + sims).prod(axis=1)

    fig, ax = fig_dark(10, 4)
    ax.hist(pat_fin/1000, bins=80, color='#3b82f6', alpha=0.6, density=True)
    ax.axvline(capital/1000, color='white', lw=1.5, ls='--', label='Capital inicial')
    p5 = float(np.percentile(pat_fin/1000, 5))
    ax.axvline(p5, color='#ef4444', lw=2, label=f'Percentil 5%: R${p5:.0f}k')
    ax.set_xlabel("Patrimônio Final (R$ mil)"); ax.set_ylabel("Densidade")
    ax.set_title(f"Monte Carlo — {N_SIM} cenários de crise", color='#94a3b8', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#334155', labelcolor='#94a3b8')
    style_ax(ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    prob = float((pat_fin < capital).mean()*100)
    perda_m = float((capital - pat_fin[pat_fin<capital]).mean()/1000) if (pat_fin<capital).any() else 0
    st.markdown(f"""<div class="alert-warn">
      Probabilidade de perda em crise: <strong>{prob:.1f}%</strong> ·
      Perda média nos cenários negativos: <strong>R$ {perda_m:.1f}k</strong>
    </div>""", unsafe_allow_html=True)
