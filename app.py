# -*- coding: utf-8 -*-
"""
=====================================================================
 MÉTRICAS DE VALUACIÓN DE ACTIVOS FINANCIEROS
 Mercados Financieros · Renta Variable
---------------------------------------------------------------------
 Dashboard en Streamlit que calcula, a partir de precios de Yahoo
 Finance, los indicadores de desempeño de activos financieros:

   Retorno anualizado ....... (Vf / Vi)^(1/n) - 1
   Volatilidad anualizada ... sigma * sqrt(n)
   Índice Sharpe ............ (Rp - Rf) / sigma_p
   Correlación de Pearson ... corr(Ri, Rm)
   Beta ..................... Cov(Ri, Rm) / var(Rm)
   Índice Traynor ........... (Ra - Rf) / beta_a
   CAPM ..................... Rf + beta_i * (Rm - Rf)
   Alpha .................... Ri - [Rf + beta_i * (Rm - Rf)]
   Valor z .................. z_alpha de la distribución normal
   VaR % / VaR $ ............ VaR_alpha = mu + z_alpha * sigma
=====================================================================
"""

from __future__ import annotations

import datetime as dt
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from scipy import stats

# =====================================================================
# 1. CONFIGURACIÓN Y TEMA VISUAL  (azul / negro · tipografía gris Arial)
# =====================================================================

st.set_page_config(
    page_title="Métricas de Valuación de Activos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Paleta (validada para superficie oscura) --------------------------
SURFACE       = "#0E1117"   # negro bursátil
SURFACE_2     = "#161B25"   # tarjetas
SURFACE_3     = "#1E2531"   # bordes
TEXT_PRIMARY  = "#E6E6E6"   # gris claro
TEXT_SECOND   = "#A3A9B5"   # gris medio
TEXT_MUTED    = "#6E7684"   # gris tenue
BLUE          = "#3987E5"   # azul principal  (serie 1)
BLUE_SOFT     = "#86B6EF"
BLUE_DEEP     = "#184F95"
ORANGE        = "#D95926"   # serie 2
AQUA          = "#199E70"   # serie 3
POS           = "#199E70"   # estatus favorable
NEG           = "#E66767"   # estatus adverso
GRID          = "#242B38"

CMAP_AZUL = LinearSegmentedColormap.from_list("azul_bursatil", ["#101A2B", BLUE_DEEP, BLUE])
SERIES = [BLUE, ORANGE, AQUA, "#C98500", "#D55181", "#9085E9", "#E66767", "#86B6EF"]
FONT = "Arial, Helvetica, sans-serif"

st.markdown(
    f"""
    <style>
      html, body, [class*="css"], .stApp {{
          font-family: {FONT};
          color: {TEXT_PRIMARY};
      }}
      .stApp {{ background: {SURFACE}; }}
      section[data-testid="stSidebar"] {{
          background: {SURFACE_2};
          border-right: 1px solid {SURFACE_3};
      }}
      section[data-testid="stSidebar"] * {{ font-family: {FONT}; color: {TEXT_SECOND}; }}
      h1, h2, h3, h4, h5 {{ font-family: {FONT}; color: {TEXT_PRIMARY}; font-weight: 600; }}

      .hdr {{
          border-left: 3px solid {BLUE};
          padding: 0.15rem 0 0.15rem 0.9rem;
          margin: 0 0 1.1rem 0;
      }}
      .hdr h1 {{ font-size: 1.65rem; margin: 0; letter-spacing: .3px; }}
      .hdr p  {{ font-size: .82rem; color: {TEXT_MUTED}; margin: .25rem 0 0 0;
                 text-transform: uppercase; letter-spacing: 1.6px; }}

      .kpi {{
          background: {SURFACE_2};
          border: 1px solid {SURFACE_3};
          border-top: 2px solid {BLUE};
          border-radius: 6px;
          padding: .85rem 1rem;
          height: 100%;
      }}
      .kpi .lbl {{ font-size: .68rem; color: {TEXT_MUTED};
                   text-transform: uppercase; letter-spacing: 1.2px; }}
      .kpi .val {{ font-size: 1.5rem; color: {TEXT_PRIMARY}; font-weight: 600;
                   line-height: 1.6rem; margin-top: .3rem; }}
      .kpi .sub {{ font-size: .72rem; color: {TEXT_SECOND}; margin-top: .25rem; }}

      .note {{ font-size: .76rem; color: {TEXT_MUTED}; line-height: 1.35rem; }}

      .stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {SURFACE_3}; }}
      .stTabs [data-baseweb="tab"] {{
          font-family: {FONT}; font-size: .82rem; letter-spacing: .8px;
          text-transform: uppercase; color: {TEXT_MUTED};
      }}
      .stTabs [aria-selected="true"] {{ color: {BLUE}; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {SURFACE_3}; border-radius: 6px; }}
      .stButton>button, .stDownloadButton>button {{
          background: {BLUE}; color: #FFFFFF; border: 0; border-radius: 4px;
          font-family: {FONT}; font-weight: 600; letter-spacing: .4px;
      }}
      #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def plotly_layout(fig: go.Figure, height: int = 420, **kwargs) -> go.Figure:
    """Aplica el tema bursátil a cualquier figura."""
    kwargs.setdefault("title", fig.layout.title.text or "")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE_2,
        plot_bgcolor=SURFACE_2,
        font=dict(family=FONT, color=TEXT_SECOND, size=12),
        title_font=dict(family=FONT, color=TEXT_PRIMARY, size=15),
        height=height,
        margin=dict(l=60, r=30, t=55, b=50),
        hoverlabel=dict(font=dict(family=FONT, size=12),
                        bgcolor=SURFACE, bordercolor=SURFACE_3),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID,
                     linecolor=SURFACE_3, tickfont=dict(size=11))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID,
                     linecolor=SURFACE_3, tickfont=dict(size=11))
    return fig


def kpi(label: str, value: str, sub: str = "") -> str:
    return (f'<div class="kpi"><div class="lbl">{label}</div>'
            f'<div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


# =====================================================================
# 2. CATÁLOGOS
# =====================================================================

# Índices bursátiles de referencia (ticker Yahoo Finance -> país)
INDICES = {
    "S&P 500 (^GSPC) · EE.UU.":            ("^GSPC", "Estados Unidos"),
    "Nasdaq 100 (^NDX) · EE.UU.":          ("^NDX",  "Estados Unidos"),
    "Dow Jones (^DJI) · EE.UU.":           ("^DJI",  "Estados Unidos"),
    "Russell 2000 (^RUT) · EE.UU.":        ("^RUT",  "Estados Unidos"),
    "S&P/BMV IPC (^MXX) · México":         ("^MXX",  "México"),
    "S&P/TSX (^GSPTSE) · Canadá":          ("^GSPTSE", "Canadá"),
    "IBEX 35 (^IBEX) · España":            ("^IBEX", "España"),
    "EURO STOXX 50 (^STOXX50E) · Zona €":  ("^STOXX50E", "Zona Euro"),
    "DAX (^GDAXI) · Alemania":             ("^GDAXI", "Alemania"),
    "CAC 40 (^FCHI) · Francia":            ("^FCHI", "Francia"),
    "FTSE 100 (^FTSE) · Reino Unido":      ("^FTSE", "Reino Unido"),
    "Nikkei 225 (^N225) · Japón":          ("^N225", "Japón"),
    "IBOVESPA (^BVSP) · Brasil":           ("^BVSP", "Brasil"),
    "S&P/CLX IPSA (^IPSA) · Chile":        ("^IPSA", "Chile"),
    "Hang Seng (^HSI) · Hong Kong":        ("^HSI",  "Hong Kong"),
    "Otro (escribir ticker)":              ("",      "Personalizado"),
}

# Tasa libre de riesgo de referencia por país (valores editables por el usuario).
# Se toma el instrumento soberano de corto plazo del país de origen del activo.
RF_PAIS = {
    "Estados Unidos": (4.20, "T-Bill 13 semanas (^IRX) / Nota 10 años (^TNX)"),
    "México":         (7.75, "CETES 28 días (Banxico)"),
    "Canadá":         (3.00, "Government of Canada 3M"),
    "Zona Euro":      (2.25, "Euribor 3M / Bund 10A"),
    "España":         (2.50, "Letra del Tesoro 12M"),
    "Alemania":       (2.20, "Bund 10 años"),
    "Francia":        (2.60, "OAT 10 años"),
    "Reino Unido":    (4.00, "Gilt 10 años"),
    "Japón":          (1.00, "JGB 10 años"),
    "Brasil":         (10.50, "Tesouro Selic"),
    "Chile":          (5.00, "BCP Banco Central de Chile"),
    "Colombia":       (8.50, "TES corto plazo"),
    "Argentina":      (30.00, "LECAP"),
    "Perú":           (5.00, "BCRP corto plazo"),
    "Hong Kong":      (3.80, "HIBOR 3M"),
    "Personalizado":  (0.00, "Definida por el usuario"),
}

# Adjetivo gramatical de la periodicidad para las etiquetas
ADJ_PERIODO = {"Diaria": "diario", "Semanal": "semanal", "Mensual": "mensual"}

# Periodicidad -> (regla de resample, periodos por año, días naturales por periodo)
PERIODICIDAD = {
    "Diaria":  ("B",     252, 1),
    "Semanal": ("W-FRI", 52,  7),
    "Mensual": ("ME",    12,  30),
}

# Plazo a calcular -> periodo de descarga
PLAZOS = {
    "5 días":   dict(days=7),
    "3 meses":  dict(days=91),
    "6 meses":  dict(days=182),
    "12 meses": dict(days=365),
    "1 año":    dict(days=365),
    "5 años":   dict(days=1826),
}

NIVELES_CONFIANZA = {"90%": 0.90, "95%": 0.95, "97.5%": 0.975, "99%": 0.99}


# =====================================================================
# 3. DESCARGA DE DATOS
# =====================================================================

@st.cache_data(ttl=900, show_spinner=False)
def descargar_precios(tickers: tuple[str, ...], inicio: dt.date, fin: dt.date,
                      ajustado: bool) -> pd.DataFrame:
    """Descarga precios de cierre (o cierre ajustado) desde Yahoo Finance."""
    data = yf.download(
        list(tickers),
        start=inicio,
        end=fin,
        auto_adjust=ajustado,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data is None or len(data) == 0:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        nivel = "Close" if "Close" in data.columns.get_level_values(0) else data.columns[0][0]
        precios = data[nivel].copy()
    else:
        col = "Close" if "Close" in data.columns else data.columns[-1]
        precios = data[[col]].copy()
        precios.columns = [tickers[0]]

    precios = precios.reindex(columns=[t for t in tickers if t in precios.columns])
    precios.index = pd.to_datetime(precios.index).tz_localize(None)
    return precios.dropna(how="all").ffill()


@st.cache_data(ttl=3600, show_spinner=False)
def tasa_libre_riesgo_auto(ticker: str) -> float | None:
    """Obtiene el último rendimiento soberano publicado en Yahoo Finance (%)."""
    try:
        serie = yf.download(ticker, period="1mo", progress=False, auto_adjust=False)
        if serie is None or serie.empty:
            return None
        col = "Close" if "Close" in serie.columns.get_level_values(0) else serie.columns[0]
        valor = float(pd.DataFrame(serie[col]).dropna().iloc[-1].iloc[0])
        return round(valor, 3)
    except Exception:
        return None


# =====================================================================
# 4. MOTOR DE CÁLCULO  (fórmulas del documento anexo)
# =====================================================================

@dataclass
class Parametros:
    rf: float                 # tasa libre de riesgo anual (decimal)
    ppa: int                  # periodos por año (252 / 52 / 12)
    dias_periodo: int         # días naturales por periodo
    confianza: float          # 0.95, 0.99, ...
    horizonte: int            # plazo del VaR en días
    capital: float            # monto de capital a invertir


def retorno_anualizado(precios: pd.Series) -> float:
    """Retorno Anual = (Valor Final / Valor Inicial)^(1/n) - 1 ; n = número de años."""
    serie = precios.dropna()
    if len(serie) < 2:
        return np.nan
    vi, vf = float(serie.iloc[0]), float(serie.iloc[-1])
    if vi <= 0:
        return np.nan
    n_anios = (serie.index[-1] - serie.index[0]).days / 365.25
    if n_anios <= 0:
        return np.nan
    return (vf / vi) ** (1.0 / n_anios) - 1.0


def volatilidad_anualizada(retornos: pd.Series, ppa: int) -> float:
    """Volatilidad Anual = sigma * sqrt(n) ; n = periodos en un año."""
    serie = retornos.dropna()
    if len(serie) < 2:
        return np.nan
    return float(serie.std(ddof=1)) * np.sqrt(ppa)


def beta_activo(r_act: pd.Series, r_mkt: pd.Series) -> float:
    """Beta = Cov(Ri, Rm) / var(Rm)."""
    df = pd.concat([r_act, r_mkt], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    var_m = float(df.iloc[:, 1].var(ddof=1))
    if var_m == 0:
        return np.nan
    cov = float(np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)[0, 1])
    return cov / var_m


def correlacion_pearson(r_act: pd.Series, r_mkt: pd.Series) -> float:
    df = pd.concat([r_act, r_mkt], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def calcular_metricas(precios: pd.DataFrame, retornos: pd.DataFrame,
                      activos: list[str], indice: str, p: Parametros) -> pd.DataFrame:
    """Construye la tabla de indicadores para cada activo."""
    z = float(stats.norm.ppf(1.0 - p.confianza))          # valor z (negativo)
    factor = p.horizonte / p.dias_periodo                  # escalamiento del VaR
    rm = retorno_anualizado(precios[indice])               # retorno del mercado
    r_mkt = retornos[indice]

    filas = []
    for a in activos:
        r_a = retornos[a]
        ret_an = retorno_anualizado(precios[a])
        vol_an = volatilidad_anualizada(r_a, p.ppa)
        beta   = beta_activo(r_a, r_mkt)
        corr   = correlacion_pearson(r_a, r_mkt)

        sharpe  = (ret_an - p.rf) / vol_an if vol_an and not np.isnan(vol_an) else np.nan
        traynor = (ret_an - p.rf) / beta if beta not in (0, None) and not np.isnan(beta) else np.nan
        capm    = p.rf + beta * (rm - p.rf) if not np.isnan(beta) else np.nan
        alpha   = ret_an - capm if not np.isnan(capm) else np.nan

        mu_h    = float(r_a.dropna().mean()) * factor
        sig_h   = float(r_a.dropna().std(ddof=1)) * np.sqrt(factor)
        var_dec = mu_h + z * sig_h                         # VaR_alpha = mu + z*sigma
        var_pct = abs(min(var_dec, 0.0))
        var_mon = var_pct * p.capital

        filas.append({
            "Activo": a,
            "Rentabilidad anualizada": ret_an,
            "Volatilidad anualizada": vol_an,
            "iSharpe": sharpe,
            "Correl. Pearson": corr,
            "BETA": beta,
            "iTraynor": traynor,
            "CAPM": capm,
            "Alpha": alpha,
            "Valor z": z,
            "VaR %": var_pct,
            "VaR $": var_mon,
        })
    return pd.DataFrame(filas).set_index("Activo")


def metricas_portafolio(precios: pd.DataFrame, retornos: pd.DataFrame,
                        activos: list[str], indice: str,
                        pesos: np.ndarray, p: Parametros) -> dict:
    """Métricas del portafolio ponderado por los pesos definidos."""
    z = float(stats.norm.ppf(1.0 - p.confianza))
    factor = p.horizonte / p.dias_periodo
    r_port = (retornos[activos] * pesos).sum(axis=1)
    r_mkt = retornos[indice]

    ret_an = (1.0 + r_port.mean()) ** p.ppa - 1.0
    vol_an = volatilidad_anualizada(r_port, p.ppa)
    beta   = beta_activo(r_port, r_mkt)
    corr   = correlacion_pearson(r_port, r_mkt)
    rm     = retorno_anualizado(precios[indice])
    capm   = p.rf + beta * (rm - p.rf) if not np.isnan(beta) else np.nan

    mu_h  = float(r_port.mean()) * factor
    sig_h = float(r_port.std(ddof=1)) * np.sqrt(factor)
    var_dec = mu_h + z * sig_h
    var_pct = abs(min(var_dec, 0.0))

    return {
        "Rentabilidad anualizada": ret_an,
        "Volatilidad anualizada": vol_an,
        "iSharpe": (ret_an - p.rf) / vol_an if vol_an else np.nan,
        "Correl. Pearson": corr,
        "BETA": beta,
        "iTraynor": (ret_an - p.rf) / beta if beta else np.nan,
        "CAPM": capm,
        "Alpha": ret_an - capm if not np.isnan(capm) else np.nan,
        "Valor z": z,
        "VaR %": var_pct,
        "VaR $": var_pct * p.capital,
        "serie": r_port,
    }


# =====================================================================
# 5. PANEL DE CONTROL (INPUTS)
# =====================================================================

st.markdown(
    '<div class="hdr"><h1>Métricas de Valuación de Activos</h1>'
    '<p>Mercados Financieros · Renta Variable · Desempeño y Riesgo</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Parámetros")

    st.markdown("**1 · Activos**")
    n_activos = st.number_input("Número de activos a valuar", 1, 8, 3, 1)
    default_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM"]
    tickers_in: list[str] = []
    cols = st.columns(2)
    for i in range(int(n_activos)):
        with cols[i % 2]:
            t = st.text_input(f"Ticker {i + 1}", default_tickers[i], key=f"tk{i}")
            if t.strip():
                tickers_in.append(t.strip().upper())
    activos = list(dict.fromkeys(tickers_in))

    st.markdown("**2 · Índice bursátil de referencia**")
    idx_label = st.selectbox("Benchmark", list(INDICES.keys()), index=0)
    idx_ticker, pais_idx = INDICES[idx_label]
    if not idx_ticker:
        idx_ticker = st.text_input("Ticker del índice", "^GSPC").strip().upper()

    st.markdown("**3 · Tasa libre de riesgo**")
    pais = st.selectbox(
        "País de origen de los activos",
        list(RF_PAIS.keys()),
        index=list(RF_PAIS.keys()).index(pais_idx) if pais_idx in RF_PAIS else 0,
    )
    rf_ref, rf_fuente = RF_PAIS[pais]
    if pais == "Estados Unidos":
        instr = st.radio("Instrumento", ["T-Bill 13s (^IRX)", "Nota 10 años (^TNX)", "Manual"],
                         horizontal=False, index=0)
        if instr != "Manual":
            tk_rf = "^IRX" if "IRX" in instr else "^TNX"
            auto = tasa_libre_riesgo_auto(tk_rf)
            if auto is not None:
                rf_ref = auto
                st.caption(f"Último dato {tk_rf}: {auto:.3f}%")
    rf_pct = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 100.0,
                             float(rf_ref), 0.05, format="%.3f")
    st.caption(f"Referencia: {rf_fuente}")

    st.markdown("**4 · Ventana de análisis**")
    plazo = st.selectbox("Plazo a calcular", list(PLAZOS.keys()), index=3)
    periodicidad = st.selectbox("Periodicidad de precios", list(PERIODICIDAD.keys()), index=0)
    tipo_precio = st.radio("Serie de precios", ["Cierre ajustado", "Cierre"],
                           horizontal=True, index=0)

    st.markdown("**5 · Valor en Riesgo (VaR)**")
    capital = st.number_input("Monto de capital a invertir ($)",
                              1000.0, 1_000_000_000.0, 1_000_000.0, 1000.0, format="%.2f")
    conf_label = st.selectbox("Intervalo de confianza", list(NIVELES_CONFIANZA.keys()), index=1)
    horizonte = st.select_slider("Plazo para VaR (días)",
                                 options=[1, 5, 10, 21, 63, 126, 252], value=10)

    st.markdown("**6 · Ponderación del portafolio**")
    equal_w = st.checkbox("Pesos iguales", value=True)
    if equal_w or not activos:
        pesos = np.repeat(1.0 / max(len(activos), 1), max(len(activos), 1))
    else:
        crudos = [st.number_input(f"Peso {a} (%)", 0.0, 100.0,
                                  round(100 / len(activos), 2), 0.5, key=f"w{a}")
                  for a in activos]
        total = sum(crudos)
        pesos = np.array(crudos) / total if total > 0 else np.repeat(1 / len(activos), len(activos))

    calcular = st.button("Calcular métricas", width="stretch")

if not activos:
    st.info("Captura al menos un ticker en el panel lateral para iniciar el análisis.")
    st.stop()

# =====================================================================
# 6. OBTENCIÓN Y PREPARACIÓN DE DATOS
# =====================================================================

regla, ppa, dias_periodo = PERIODICIDAD[periodicidad]
fin = dt.date.today() + dt.timedelta(days=1)
inicio = fin - dt.timedelta(**PLAZOS[plazo])
universo = tuple(activos + [idx_ticker])

with st.spinner("Descargando precios de Yahoo Finance…"):
    precios_raw = descargar_precios(universo, inicio, fin, tipo_precio == "Cierre ajustado")

if precios_raw.empty:
    st.error("No se obtuvieron precios. Verifica los tickers y el índice de referencia.")
    st.stop()

faltantes = [t for t in universo if t not in precios_raw.columns]
if faltantes:
    st.warning(f"Sin datos para: {', '.join(faltantes)}. Se excluyen del análisis.")
activos = [a for a in activos if a in precios_raw.columns]
if idx_ticker not in precios_raw.columns or not activos:
    st.error("Se requiere al menos un activo y el índice de referencia con datos válidos.")
    st.stop()

precios = precios_raw.resample(regla).last().dropna(how="all").ffill().dropna()
retornos = precios.pct_change().dropna()

if len(retornos) < 5:
    st.warning(
        f"Solo hay {len(retornos)} observaciones con periodicidad **{periodicidad.lower()}** "
        f"en un plazo de **{plazo}**. Los estadísticos serán poco robustos: "
        "usa una periodicidad más fina o amplía el plazo."
    )
if len(retornos) < 3:
    st.stop()

par = Parametros(
    rf=rf_pct / 100.0, ppa=ppa, dias_periodo=dias_periodo,
    confianza=NIVELES_CONFIANZA[conf_label], horizonte=int(horizonte), capital=float(capital),
)

tabla = calcular_metricas(precios, retornos, activos, idx_ticker, par)
port = metricas_portafolio(precios, retornos, activos, idx_ticker, pesos, par)

# =====================================================================
# 7. ENCABEZADO DE RESULTADOS (KPIs del portafolio)
# =====================================================================

st.caption(
    f"{len(activos)} activo(s) · benchmark {idx_ticker} · {periodicidad.lower()} · "
    f"{precios.index[0]:%d-%b-%Y} → {precios.index[-1]:%d-%b-%Y} · "
    f"{len(retornos)} observaciones · Rf {rf_pct:.2f}%"
)

k = st.columns(5)
with k[0]:
    st.markdown(kpi("Rentabilidad portafolio",
                    f"{port['Rentabilidad anualizada']:.2%}",
                    "Ponderada por los pesos definidos"), unsafe_allow_html=True)
with k[1]:
    st.markdown(kpi("Volatilidad anualizada",
                    f"{port['Volatilidad anualizada']:.2%}",
                    f"σ × √{ppa}"), unsafe_allow_html=True)
with k[2]:
    st.markdown(kpi("Índice Sharpe", f"{port['iSharpe']:.2f}",
                    "Retorno por unidad de riesgo total"), unsafe_allow_html=True)
with k[3]:
    st.markdown(kpi("BETA vs índice", f"{port['BETA']:.2f}",
                    f"Correlación {port['Correl. Pearson']:.2f}"), unsafe_allow_html=True)
with k[4]:
    st.markdown(kpi(f"VaR {conf_label} · {horizonte}d",
                    f"${port['VaR $']:,.0f}",
                    f"{port['VaR %']:.2%} del capital · z = {port['Valor z']:.3f}"),
                unsafe_allow_html=True)

st.write("")

# =====================================================================
# 8. PESTAÑAS DE ANÁLISIS
# =====================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    ["Indicadores", "Correlación", "Regresión vs índice", "Precios y VaR"]
)

FORMATOS = {
    "Rentabilidad anualizada": "{:.2%}", "Volatilidad anualizada": "{:.2%}",
    "iSharpe": "{:.3f}", "Correl. Pearson": "{:.3f}", "BETA": "{:.3f}",
    "iTraynor": "{:.3f}", "CAPM": "{:.2%}", "Alpha": "{:.2%}",
    "Valor z": "{:.3f}", "VaR %": "{:.2%}", "VaR $": "${:,.2f}",
}

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.markdown("#### Tabla de indicadores de desempeño")
    vista = tabla.copy()
    vista.loc["PORTAFOLIO"] = {c: port[c] for c in tabla.columns}
    st.dataframe(
        vista.style.format(FORMATOS)
        .background_gradient(subset=["iSharpe"], cmap=CMAP_AZUL)
        .map(lambda v: f"color: {POS}" if v > 0 else f"color: {NEG}",
             subset=["Alpha", "Rentabilidad anualizada"])
        .set_properties(**{"font-family": FONT}),
        width="stretch",
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tabla.index, y=tabla["Rentabilidad anualizada"] * 100,
            marker_color=BLUE, marker_line_width=0, width=0.55,
            text=[f"{v:.1%}" for v in tabla["Rentabilidad anualizada"]],
            textposition="outside", textfont=dict(color=TEXT_SECOND, size=11),
            hovertemplate="<b>%{x}</b><br>Rentabilidad anual: %{y:.2f}%<extra></extra>",
            name="Rentabilidad",
        ))
        fig.add_hline(y=rf_pct, line_color=ORANGE, line_dash="dot", line_width=2)
        fig.add_annotation(xref="paper", yref="paper", x=1, y=1.03, xanchor="right",
                           showarrow=False, text=f"⋯ Tasa libre de riesgo {rf_pct:.2f}%",
                           font=dict(family=FONT, size=11, color=ORANGE))
        plotly_layout(fig, 380, title="Rentabilidad anualizada por activo",
                      yaxis_title="%", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tabla["Volatilidad anualizada"] * 100,
            y=tabla["Rentabilidad anualizada"] * 100,
            mode="markers+text", text=tabla.index, textposition="top center",
            textfont=dict(color=TEXT_SECOND, size=11),
            marker=dict(size=15, color=BLUE, line=dict(color=SURFACE_2, width=2)),
            hovertemplate=("<b>%{text}</b><br>Volatilidad: %{x:.2f}%"
                           "<br>Rentabilidad: %{y:.2f}%<extra></extra>"),
            name="Activos",
        ))
        fig.add_trace(go.Scatter(
            x=[port["Volatilidad anualizada"] * 100],
            y=[port["Rentabilidad anualizada"] * 100],
            mode="markers+text", text=["PORTAFOLIO"], textposition="bottom center",
            textfont=dict(color=ORANGE, size=11),
            marker=dict(size=16, color=ORANGE, symbol="diamond",
                        line=dict(color=SURFACE_2, width=2)),
            hovertemplate="<b>Portafolio</b><br>Vol: %{x:.2f}%<br>Rent: %{y:.2f}%<extra></extra>",
            name="Portafolio",
        ))
        plotly_layout(fig, 380, title="Riesgo vs. rentabilidad",
                      xaxis_title="Volatilidad anualizada (%)",
                      yaxis_title="Rentabilidad anualizada (%)", showlegend=False)
        st.plotly_chart(fig, width="stretch")

    buffer = io.StringIO()
    vista.to_csv(buffer)
    st.download_button("Descargar indicadores (CSV)", buffer.getvalue(),
                       file_name="metricas_valuacion_activos.csv", mime="text/csv")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.markdown("#### Matriz de correlación de Pearson")
    corr = retornos[activos + [idx_ticker]].corr()
    etiquetas = [[f"{v:.2f}" for v in fila] for fila in corr.values]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale=[[0.0, "#E66767"], [0.5, "#383835"], [1.0, BLUE]],
        text=etiquetas, texttemplate="%{text}",
        textfont=dict(family=FONT, size=12, color=TEXT_PRIMARY),
        xgap=2, ygap=2,
        hovertemplate="%{y} ↔ %{x}<br>ρ = %{z:.3f}<extra></extra>",
        colorbar=dict(title=dict(text="ρ", font=dict(color=TEXT_SECOND)),
                      tickfont=dict(color=TEXT_SECOND), outlinewidth=0, thickness=14),
    ))
    plotly_layout(fig, 120 + 58 * len(corr),
                  title=f"Correlación de retornos ({ADJ_PERIODO[periodicidad]}s)")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        '<p class="note">Azul = comovimiento positivo · Gris = independencia · '
        'Rojo = comovimiento inverso. Correlaciones bajas entre activos indican '
        'mayor beneficio de diversificación.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.markdown(f"#### Regresión de cada activo contra {idx_ticker}")
    ncol = min(3, len(activos))
    nfil = int(np.ceil(len(activos) / ncol))
    fig = make_subplots(
        rows=nfil, cols=ncol, shared_yaxes=False,
        subplot_titles=[f"{a} vs {idx_ticker}" for a in activos],
        horizontal_spacing=0.09, vertical_spacing=0.18,
    )
    x = retornos[idx_ticker]
    for i, a in enumerate(activos):
        fila, col = i // ncol + 1, i % ncol + 1
        y = retornos[a]
        df = pd.concat([x, y], axis=1).dropna()
        pend, ordn, r_val, _, _ = stats.linregress(df.iloc[:, 0], df.iloc[:, 1])
        xs = np.linspace(df.iloc[:, 0].min(), df.iloc[:, 0].max(), 60)

        fig.add_trace(go.Scatter(
            x=df.iloc[:, 0] * 100, y=df.iloc[:, 1] * 100, mode="markers",
            marker=dict(size=7, color=BLUE, opacity=0.55,
                        line=dict(color=SURFACE_2, width=1)),
            name=a, showlegend=False,
            hovertemplate=(f"<b>{a}</b><br>{idx_ticker}: %{{x:.2f}}%"
                           f"<br>{a}: %{{y:.2f}}%<extra></extra>"),
        ), row=fila, col=col)
        fig.add_trace(go.Scatter(
            x=xs * 100, y=(ordn + pend * xs) * 100, mode="lines",
            line=dict(color=ORANGE, width=2), name="Ajuste OLS",
            showlegend=False, hoverinfo="skip",
        ), row=fila, col=col)
        sub = "" if i == 0 else str(i + 1)
        fig.add_annotation(
            row=fila, col=col, xref=f"x{sub} domain", yref=f"y{sub} domain",
            x=0.03, y=0.97, showarrow=False, align="left",
            text=(f"β = {pend:.3f}<br>α<sub>periodo</sub> = {ordn:.4%}"
                  f"<br>R² = {r_val ** 2:.3f}<br>ρ = {r_val:.3f}"),
            font=dict(family=FONT, size=11, color=TEXT_SECOND),
            bgcolor=SURFACE, bordercolor=SURFACE_3, borderwidth=1, borderpad=6,
        )
    for ann in fig.layout.annotations[:len(activos)]:
        ann.font = dict(family=FONT, size=12, color=TEXT_PRIMARY)
    plotly_layout(fig, 330 * nfil, title="")
    fig.update_xaxes(title_text=f"Retorno {idx_ticker} (%)", title_font=dict(size=11))
    fig.update_yaxes(title_text="Retorno activo (%)", title_font=dict(size=11))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        '<p class="note">La pendiente de la recta es la <b>BETA</b> del activo '
        '(sensibilidad al mercado); la ordenada al origen es el <b>alpha</b> del '
        'periodo y R² indica qué proporción del movimiento del activo explica el índice.</p>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------- Tab 4
with tab4:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("#### Evolución de precios (base 100)")
        base = precios[activos + [idx_ticker]] / precios[activos + [idx_ticker]].iloc[0] * 100
        fig = go.Figure()
        for i, cnm in enumerate(activos):
            fig.add_trace(go.Scatter(
                x=base.index, y=base[cnm], mode="lines", name=cnm,
                line=dict(color=SERIES[i % len(SERIES)], width=2),
                hovertemplate=f"<b>{cnm}</b><br>%{{x|%d-%b-%Y}}<br>%{{y:.1f}}<extra></extra>",
            ))
        fig.add_trace(go.Scatter(
            x=base.index, y=base[idx_ticker], mode="lines", name=idx_ticker,
            line=dict(color=TEXT_MUTED, width=2, dash="dash"),
            hovertemplate=f"<b>{idx_ticker}</b><br>%{{x|%d-%b-%Y}}<br>%{{y:.1f}}<extra></extra>",
        ))
        plotly_layout(fig, 420, yaxis_title="Índice base 100", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown(f"#### VaR {conf_label} a {horizonte} días")
        var_df = tabla[["VaR %", "VaR $"]].copy()
        var_df.loc["PORTAFOLIO"] = [port["VaR %"], port["VaR $"]]
        fig = go.Figure(go.Bar(
            x=var_df["VaR $"], y=var_df.index, orientation="h",
            marker_color=[ORANGE if n == "PORTAFOLIO" else BLUE for n in var_df.index],
            marker_line_width=0,
            text=[f"${v:,.0f}" for v in var_df["VaR $"]],
            textposition="outside", textfont=dict(color=TEXT_SECOND, size=11),
            hovertemplate="<b>%{y}</b><br>Pérdida máxima esperada: $%{x:,.2f}<extra></extra>",
        ))
        plotly_layout(fig, 420, xaxis_title="Pérdida máxima esperada ($)", showlegend=False)
        fig.update_xaxes(range=[0, float(var_df["VaR $"].max()) * 1.28])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Distribución de retornos del portafolio")
    z = port["Valor z"]
    serie = port["serie"].dropna() * 100
    corte = (serie.mean() + z * serie.std(ddof=1))
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=serie, nbinsx=45, marker_color=BLUE_DEEP,
        marker_line=dict(color=SURFACE_2, width=2), name="Retornos",
        hovertemplate="Retorno: %{x:.2f}%<br>Frecuencia: %{y}<extra></extra>",
    ))
    fig.add_vline(x=corte, line_color=NEG, line_width=2, line_dash="dash",
                  annotation_text=f"VaR {conf_label} {ADJ_PERIODO[periodicidad]}: {corte:.2f}%",
                  annotation_font=dict(color=NEG, size=11))
    plotly_layout(fig, 330, xaxis_title=f"Retorno {ADJ_PERIODO[periodicidad]} (%)",
                  yaxis_title="Frecuencia", showlegend=False, bargap=0.04)
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f'<p class="note">VaR<sub>α</sub> = μ + z<sub>α</sub>·σ, escalado a '
        f'{horizonte} día(s) mediante μ·t y σ·√t. Con {conf_label} de confianza, '
        f'la pérdida del portafolio no debería exceder '
        f'<b>${port["VaR $"]:,.2f}</b> ({port["VaR %"]:.2%} de ${capital:,.0f}) '
        f'en el horizonte definido.</p>', unsafe_allow_html=True)

# =====================================================================
# 9. NOTA METODOLÓGICA
# =====================================================================

with st.expander("Fórmulas y notas metodológicas"):
    st.markdown(
        f"""
| Indicador | Fórmula | Implementación |
|---|---|---|
| Retorno anual | $(V_f/V_i)^{{1/n}}-1$ | $n$ = años naturales de la ventana |
| Volatilidad anual | $\\sigma\\sqrt{{n}}$ | $n$ = {ppa} periodos/año ({ADJ_PERIODO[periodicidad]}) |
| Índice Sharpe | $(R_p-R_f)/\\sigma_p$ | Ambos términos anualizados |
| Beta | $\\mathrm{{Cov}}(R_i,R_m)/\\sigma_m^2$ | Retornos del periodo seleccionado |
| Índice Traynor | $(R_a-R_f)/\\beta_a$ | Riesgo sistemático |
| CAPM | $R_f+\\beta_i(R_m-R_f)$ | $R_m$ = retorno anualizado del índice |
| Alpha | $R_i-[R_f+\\beta_i(R_m-R_f)]$ | Exceso sobre el retorno exigido |
| VaR | $\\mu+z_\\alpha\\sigma$ | $z_\\alpha$ = {NIVELES_CONFIANZA[conf_label]:.3f} → {port['Valor z']:.4f} |

**Supuestos.** Retornos aritméticos; distribución normal para el VaR paramétrico;
Rf constante durante la ventana; sin costos de transacción, impuestos ni dividendos
en efectivo cuando se usa la serie de *Cierre* simple. En ventanas cortas (5 días,
3 meses) la anualización amplifica el ruido: interpreta esos resultados como
indicativos.

**Fuente de datos.** Yahoo Finance vía `yfinance`. Tasa libre de riesgo: instrumento
soberano de corto plazo del país de origen de los activos, capturable manualmente.
"""
    )

st.markdown(
    f'<p class="note" style="margin-top:1.4rem;border-top:1px solid {SURFACE_3};'
    f'padding-top:.8rem;">Herramienta con fines académicos. No constituye una '
    f'recomendación de inversión. Datos: Yahoo Finance.</p>',
    unsafe_allow_html=True,
)
