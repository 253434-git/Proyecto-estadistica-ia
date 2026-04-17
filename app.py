import streamlit as st
import pandas as pd
import numpy as np

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