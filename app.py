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

# --- MÓDULO 2: VISUALIZACIÓN ---
if df is not None:
    st.header("📈 Visualización de Distribuciones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histograma con KDE
        fig_hist = px.histogram(df, x=col_seleccionada, marginal="box", 
                                 title=f"Distribución de {col_seleccionada}",
                                 nbins=30, kde=True)
        st.plotly_chart(fig_hist)

    with col2:
        # Boxplot para Outliers
        fig_box = px.box(df, y=col_seleccionada, title=f"Boxplot de {col_seleccionada}")
        st.plotly_chart(fig_box)

    # Preguntas de interpretación (Requisito de la tarea)
    st.subheader("¿Qué observamos?")
    res_normal = st.radio("¿La distribución parece normal?", ["Sí", "No", "No estoy seguro"])
    res_sesgo = st.text_input("¿Hay sesgo o outliers? Explica brevemente:")

    