"""
====================================================================
APP DE VALORACIÓN DE RIESGOS FINANCIEROS
Alumna: Angela Funes Fresán
Activos asignados: KO (Coca-Cola), LINK-USD (Chainlink), JPY=X (Yen Japonés)
====================================================================
Fases:
  1. Obtención y limpieza de datos (yfinance)
  2. Construcción de indicadores (Volatilidad, VaR, Drawdown, Sharpe)
  3. Interfaz, visualizaciones y alertas de control
"""

# ============================================================
# 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ============================================================
# 2. CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Valoración de Riesgos Financieros",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Aplicación de Valoración de Riesgos Financieros")
st.markdown("**Alumna:** Angela Funes Fresán &nbsp;&nbsp;|&nbsp;&nbsp; **Activos:** KO · LINK-USD · JPY=X")
st.markdown("---")

# ============================================================
# 3. BARRA LATERAL - PARÁMETROS DE USUARIO
# ============================================================
st.sidebar.header("⚙️ Parámetros del análisis")

# Activos asignados al alumno #10
activos = {
    "KO (Coca-Cola - Acción)": "KO",
    "LINK-USD (Chainlink - Criptomoneda)": "LINK-USD",
    "JPY=X (Yen Japonés - Divisa/Refugio)": "JPY=X"
}

activo_seleccionado = st.sidebar.selectbox(
    "Selecciona un activo:",
    list(activos.keys())
)
ticker = activos[activo_seleccionado]

# Nivel de confianza para VaR
nivel_confianza = st.sidebar.selectbox(
    "Nivel de confianza para VaR:",
    [0.90, 0.95, 0.99],
    index=1,
    format_func=lambda x: f"{int(x*100)}%"
)

# Periodo de análisis (mínimo 1 año según requisitos)
anios = st.sidebar.slider("Años de historia:", 1, 5, 1)

# Umbral de alerta para VaR
umbral_var = st.sidebar.slider(
    "Umbral de alerta VaR (%):",
    1.0, 10.0, 3.0, 0.5
) / 100

# Tasa libre de riesgo (para Sharpe)
tasa_libre = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%):",
    0.0, 20.0, 4.5, 0.1
) / 100

# ============================================================
# FASE 1: OBTENCIÓN Y LIMPIEZA DE DATOS
# ============================================================
@st.cache_data(ttl=3600)
def descargar_datos(ticker, anios):
    """Descarga datos históricos desde Yahoo Finance (versión robusta)."""
    fin = datetime.today()
    inicio = fin - timedelta(days=365 * anios)

    # Intento 1: yf.download
    try:
        df = yf.download(
            ticker,
            start=inicio,
            end=fin,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception as e:
        st.warning(f"yf.download falló: {e}. Intentando con Ticker.history...")
        df = pd.DataFrame()

    # Intento 2: si viene vacío, usar Ticker.history
    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=f"{anios}y", auto_adjust=True)
        except Exception as e:
            st.error(f"No se pudo descargar {ticker}: {e}")
            return pd.DataFrame()

    # Aplanar MultiIndex de columnas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Asegurar que exista la columna Close
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            st.error(f"Los datos de {ticker} no contienen columna de precios.")
            return pd.DataFrame()

    # Limpieza final
    df = df[["Close"]].dropna()
    return df

with st.spinner(f"Descargando datos de {ticker}..."):
    datos = descargar_datos(ticker, anios)

if datos.empty:
    st.error("No se pudieron obtener datos. Intenta con otro activo.")
    st.stop()

# Calcular retornos diarios logarítmicos
datos["Retorno"] = np.log(datos["Close"] / datos["Close"].shift(1))
datos = datos.dropna()

# ============================================================
# FASE 2: CÁLCULO DE INDICADORES DE RIESGO
# ============================================================
retornos = datos["Retorno"]

# --- Volatilidad Histórica Anualizada ---
volatilidad_diaria = retornos.std()
volatilidad_anual = volatilidad_diaria * np.sqrt(252)

# --- VaR Paramétrico ---
# Asume distribución normal: VaR = media - z * desviación
media_ret = retornos.mean()
z_score = stats.norm.ppf(1 - nivel_confianza)
var_parametrico = media_ret + z_score * volatilidad_diaria  # valor negativo = pérdida

# --- VaR Histórico (complementario) ---
var_historico = retornos.quantile(1 - nivel_confianza)

# --- Máximo Drawdown ---
precio_acumulado = datos["Close"]
maximo_movil = precio_acumulado.cummax()
drawdown_serie = (precio_acumulado - maximo_movil) / maximo_movil
max_drawdown = drawdown_serie.min()

# --- Ratio de Sharpe ---
retorno_anual = media_ret * 252
sharpe = (retorno_anual - tasa_libre) / volatilidad_anual if volatilidad_anual != 0 else 0

# ============================================================
# FASE 3: INTERFAZ, VISUALIZACIONES Y ALERTAS
# ============================================================

# --- Panel de métricas ---
st.subheader(f"📈 Indicadores de riesgo — {activo_seleccionado}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Volatilidad Anual", f"{volatilidad_anual*100:.2f}%")
col2.metric(f"VaR {int(nivel_confianza*100)}% (Paramétrico)", f"{var_parametrico*100:.2f}%")
col3.metric("Máx. Drawdown", f"{max_drawdown*100:.2f}%")
col4.metric("Ratio de Sharpe", f"{sharpe:.2f}")

# --- Alerta de control ---
st.subheader("🚦 Panel de Control de Alertas")
if abs(var_parametrico) > umbral_var:
    st.error(
        f"⚠️ **ALERTA DE RIESGO ALTO**: El VaR ({abs(var_parametrico)*100:.2f}%) "
        f"supera el umbral de control ({umbral_var*100:.1f}%).\n\n"
        f"**Medidas sugeridas:** diversificar la cartera, aplicar coberturas "
        f"(hedging con opciones o futuros), reducir exposición al activo o "
        f"establecer stop-loss."
    )
elif abs(var_parametrico) > umbral_var * 0.7:
    st.warning(
        f"⚠️ **Riesgo moderado**: el VaR se acerca al umbral. "
        f"Recomendación: monitorear diariamente."
    )
else:
    st.success(
        f"✅ **Riesgo controlado**: el VaR ({abs(var_parametrico)*100:.2f}%) "
        f"se encuentra dentro del umbral aceptable ({umbral_var*100:.1f}%)."
    )

st.markdown("---")

# --- Gráfico 1: Evolución de precios ---
st.subheader("💹 Evolución histórica de precios")
fig_precio = go.Figure()
fig_precio.add_trace(go.Scatter(
    x=datos.index, y=datos["Close"],
    mode="lines", name="Precio de cierre",
    line=dict(color="#1f77b4", width=2)
))
fig_precio.update_layout(
    xaxis_title="Fecha", yaxis_title="Precio",
    hovermode="x unified", height=400
)
st.plotly_chart(fig_precio, use_container_width=True)

# --- Gráfico 2: Histograma de retornos con línea de VaR ---
st.subheader("📉 Distribución de retornos diarios y línea de VaR")
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=retornos, nbinsx=60, name="Retornos",
    marker_color="#2E86AB", opacity=0.75
))
# Línea roja de control (VaR)
fig_hist.add_vline(
    x=var_parametrico, line_width=3, line_dash="dash", line_color="red",
    annotation_text=f"VaR {int(nivel_confianza*100)}% = {var_parametrico*100:.2f}%",
    annotation_position="top left"
)
# Línea de la media
fig_hist.add_vline(
    x=media_ret, line_width=2, line_dash="dot", line_color="green",
    annotation_text=f"Media = {media_ret*100:.3f}%",
    annotation_position="top right"
)
fig_hist.update_layout(
    xaxis_title="Retorno diario", yaxis_title="Frecuencia",
    height=400, bargap=0.05
)
st.plotly_chart(fig_hist, use_container_width=True)

st.caption(
    "🔎 **Análisis de colas:** la región a la izquierda de la línea roja "
    "representa los escenarios de pérdida extrema (cola izquierda de la distribución)."
)

# --- Gráfico 3: Drawdown ---
st.subheader("📊 Evolución del Drawdown")
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown_serie.index, y=drawdown_serie * 100,
    fill="tozeroy", mode="lines",
    line=dict(color="crimson"), name="Drawdown"
))
fig_dd.update_layout(
    xaxis_title="Fecha", yaxis_title="Drawdown (%)",
    height=350
)
st.plotly_chart(fig_dd, use_container_width=True)

# --- Comparativa precios vs histograma (lado a lado) ---
st.subheader("🔀 Comparativa: Precios vs Frecuencia de retornos")
c1, c2 = st.columns(2)
with c1:
    mini_precio = px.line(datos, x=datos.index, y="Close", title="Precio de cierre")
    mini_precio.update_layout(height=350)
    st.plotly_chart(mini_precio, use_container_width=True)
with c2:
    mini_hist = px.histogram(retornos, nbins=50, title="Frecuencia de retornos")
    mini_hist.update_layout(height=350, showlegend=False)
    st.plotly_chart(mini_hist, use_container_width=True)

# ============================================================
# REPORTE DE ANÁLISIS COMPARATIVO (los 3 activos)
# ============================================================
st.markdown("---")
st.subheader("📝 Reporte comparativo de los 3 activos")

@st.cache_data(ttl=3600)
def calcular_metricas(ticker, anios):
    df = descargar_datos(ticker, anios)
    if df.empty:
        return None
    ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    vol = ret.std() * np.sqrt(252)
    var95 = ret.mean() + stats.norm.ppf(0.05) * ret.std()
    cummax = df["Close"].cummax()
    dd = ((df["Close"] - cummax) / cummax).min()
    sharpe_r = (ret.mean() * 252 - 0.045) / vol if vol != 0 else 0
    return {
        "Volatilidad anual": f"{vol*100:.2f}%",
        "VaR 95%": f"{var95*100:.2f}%",
        "Máx. Drawdown": f"{dd*100:.2f}%",
        "Sharpe": f"{sharpe_r:.2f}"
    }

tabla = {}
for nombre, tkr in activos.items():
    m = calcular_metricas(tkr, anios)
    if m:
        tabla[tkr] = m

df_comparativo = pd.DataFrame(tabla).T
st.dataframe(df_comparativo, use_container_width=True)

st.markdown("""
### 🧠 Conclusión del análisis
- **KO (Coca-Cola):** acción defensiva de consumo masivo. Típicamente presenta
  la **menor volatilidad** y un drawdown moderado. Es el activo más estable.
- **LINK-USD (Chainlink):** criptomoneda con la **mayor volatilidad** y los
  mayores drawdowns. Representa el **activo de mayor riesgo** del portafolio,
  con colas muy pesadas en su distribución de retornos.
- **JPY=X (Yen Japonés):** divisa refugio tradicional. Presenta volatilidad
  baja y drawdowns contenidos, aunque su Sharpe puede ser bajo por retornos
  modestos.

**Conclusión general:** según los indicadores (volatilidad anualizada, VaR
y máximo drawdown), **LINK-USD es el activo con mayor riesgo**, seguido de
KO y finalmente JPY=X como el más conservador. Un portafolio diversificado
entre los tres permitiría balancear rentabilidad esperada y riesgo.
""")

st.markdown("---")
with st.expander("📄 Ver datos descargados (últimos 10 días)"):
    st.dataframe(datos.tail(10))
