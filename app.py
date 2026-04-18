import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Asistente Estadístico", layout="wide")

st.title("Aplicación de Análisis Estadístico e IA")

# --- MÓDULO 1: CARGA DE DATOS ---
st.sidebar.header("Configuración de Datos")
opcion_datos = st.sidebar.selectbox("Origen de los datos", ["Subir CSV", "Generación Sintética"])

df = None

if opcion_datos == "Subir CSV":
    archivo = st.sidebar.file_uploader("Carga tu archivo CSV", type=["csv"])
    if archivo:
        df = pd.read_csv(archivo)
else:
    n_sintetico = st.sidebar.slider("Tamaño de muestra", 30, 1000, 100)
    if st.sidebar.button("Generar datos aleatorios"):
        data = np.random.normal(loc=50, scale=10, size=n_sintetico)
        df = pd.DataFrame(data, columns=["Variable_Objetivo"])

if df is not None:
    st.subheader("Vista previa de los datos")
    st.write(df.head())
    
    col_seleccionada = st.selectbox("Selecciona la variable para analizar", df.columns)
    st.info(f"Variable seleccionada: {col_seleccionada}")
else:
    st.warning("Por favor, carga datos o genera una muestra sintética en la barra lateral.")

# --- MÓDULO 2: VISUALIZACIÓN CORREGIDO ---
if df is not None:
    st.header("📈 Visualización de Distribuciones")
    
    col_seleccionada = st.selectbox("Selecciona la variable para analizar", df.select_dtypes(include=[np.number]).columns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma con "Rug" (pequeñas marcas abajo para ver la densidad)
        fig_hist = px.histogram(df, x=col_seleccionada, 
                                 title=f"Distribución de {col_seleccionada}",
                                 nbins=20, 
                                 marginal="rug",  # Esto ayuda a ver la densidad como un KDE
                                 color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_hist)

    with col2:
        # Boxplot para detectar outliers
        fig_box = px.box(df, y=col_seleccionada, 
                         title=f"Boxplot de {col_seleccionada}",
                         points="all", # Muestra todos los puntos para ver outliers claramente
                         color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_box)

    # --- MÓDULO 3: PRUEBAS ESTADÍSTICAS (PRUEBA Z) ---
    st.divider()
    st.header("🧪 Prueba de Hipótesis (Z-Test)")
    st.markdown("""
    **Supuestos de la prueba:**
    * Varianza poblacional conocida ($\sigma$).
    * Tamaño de muestra grande ($n \ge 30$).
    """)
    
    with st.expander("⚙️ Configuración de la Prueba Z", expanded=True):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            mu_h0 = st.number_input("Hipótesis Nula (μ0)", value=50.0)
            sigma = st.number_input("Varianza poblacional conocida (σ)", value=10.0, min_value=0.1)
        with col_p2:
            alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)
            tipo_test = st.selectbox("Tipo de prueba", ["Bilateral", "Cola izquierda", "Cola derecha"])

    # Cálculos Estadísticos
    n = len(df[col_seleccionada])
    media_muestral = df[col_seleccionada].mean()
    # Fórmula: Z = (x_barra - mu) / (sigma / sqrt(n))
    z_stat = (media_muestral - mu_h0) / (sigma / np.sqrt(n))
    
    # Cálculo de P-value y Z-Crítico según el tipo de prueba
    if tipo_test == "Bilateral":
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_critico_inf = stats.norm.ppf(alpha/2)
        z_critico_sup = stats.norm.ppf(1 - alpha/2)
        interpretacion_z = f"Región de rechazo: Z < {z_critico_inf:.2f} o Z > {z_critico_sup:.2f}"
    elif tipo_test == "Cola izquierda":
        p_val = stats.norm.cdf(z_stat)
        z_critico = stats.norm.ppf(alpha)
        interpretacion_z = f"Región de rechazo: Z < {z_critico:.2f}"
    else: # Cola derecha
        p_val = 1 - stats.norm.cdf(z_stat)
        z_critico = stats.norm.ppf(1 - alpha)
        interpretacion_z = f"Región de rechazo: Z > {z_critico:.2f}"

    # --- SALIDA DE RESULTADOS ---
    st.subheader("📊 Resultados de la Prueba")
    c1, c2, c3 = st.columns(3)
    c1.metric("Estadístico Z", f"{z_stat:.4f}")
    c2.metric("P-Value", f"{p_val:.4f}")
    c3.metric("Media Muestral", f"{media_muestral:.2f}")

    st.info(f"**Interpretación:** {interpretacion_z}")

    if p_val < alpha:
        st.error(f"**Decisión:** Se rechaza la Hipótesis Nula (H0).")
        st.write("Hay evidencia suficiente para afirmar que la media es distinta a la hipotética.")
    else:
        st.success(f"**Decisión:** No se rechaza la Hipótesis Nula (H0).")
        st.write("No hay evidencia suficiente para rechazar la igualdad de medias.")

    # --- MÓDULO 4: GRÁFICA DE LA CURVA NORMAL ---
    st.divider()
    st.subheader("📈 Curva de Distribución y Región Crítica")

    # Crear puntos para la curva normal estándar
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)
    
    fig_z = go.Figure()

    # Dibujar la curva normal
    fig_z.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Normal Estándar', line=dict(color='blue')))

    # Sombrear región de rechazo
    if tipo_test == "Bilateral":
        # Izquierda
        x_left = np.linspace(-4, z_critico_inf, 100)
        fig_z.add_trace(go.Scatter(x=x_left, y=stats.norm.pdf(x_left), fill='tozeroy', name='Rechazo', line_color='red'))
        # Derecha
        x_right = np.linspace(z_critico_sup, 4, 100)
        fig_z.add_trace(go.Scatter(x=x_right, y=stats.norm.pdf(x_right), fill='tozeroy', name='Rechazo', line_color='red', showlegend=False))
    elif tipo_test == "Cola izquierda":
        x_left = np.linspace(-4, z_critico, 100)
        fig_z.add_trace(go.Scatter(x=x_left, y=stats.norm.pdf(x_left), fill='tozeroy', name='Rechazo', line_color='red'))
    else: # Cola derecha
        x_right = np.linspace(z_critico, 4, 100)
        fig_z.add_trace(go.Scatter(x=x_right, y=stats.norm.pdf(x_right), fill='tozeroy', name='Rechazo', line_color='red'))

    # Línea del estadístico Z calculado
    fig_z.add_vline(x=z_stat, line_width=3, line_dash="dash", line_color="green", annotation_text=f"Z calculada: {z_stat:.2f}")

    fig_z.update_layout(title="Zonas de Rechazo (Rojo) vs Z Calculada (Línea Verde)", xaxis_title="Z", yaxis_title="Densidad")
    st.plotly_chart(fig_z)