import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import joblib
from io import BytesIO

# Configurar la página
st.title("Evaluación de Cláusulas mediante Árbol de Decisiones")
st.sidebar.title("Navegación")
app_mode = st.sidebar.selectbox("Selecciona una opción", ["Cargar Reportes de Auditoría", "Entrenar Modelo"])

if app_mode == "Cargar Reportes de Auditoría":
    st.header("Cargar Reportes de Auditoría")
    
    # Cargar el archivo CSV
    uploaded_report = st.file_uploader("Sube el archivo CSV con los reportes de auditoría", type="csv")
    if uploaded_report:
        # Leer el archivo CSV en un DataFrame
        reports_df = pd.read_csv(uploaded_report)
        
        # Verificar si el CSV contiene las columnas necesarias
        required_columns = ["auditoria_id", "clausula", "no_conformidad_critica", "no_conformidad_mayor", "no_conformidad_menor", "riesgo"]
        if not all(column in reports_df.columns for column in required_columns):
            st.error("El archivo CSV debe contener las columnas: " + ", ".join(required_columns))
        else:
            # Generar el CSV adecuado para el entrenamiento del árbol de decisiones
            grouped_report = reports_df.groupby("clausula").agg({
                "no_conformidad_critica": "sum",
                "no_conformidad_mayor": "sum",
                "no_conformidad_menor": "sum",
                "riesgo": "first"  # Suponemos que el riesgo es el mismo para todas las auditorías de la cláusula
            }).reset_index()

            # Descargar el CSV generado
            result_csv = grouped_report.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar CSV para Entrenamiento", result_csv, file_name="datos_entrenamiento_arbol.csv")

elif app_mode == "Entrenar Modelo":
    st.header("Entrenamiento del Modelo de Árbol de Decisiones")

    # Cargar el archivo CSV
    uploaded_file = st.file_uploader("Sube el archivo CSV con las columnas: `auditoria_id`, `clausula`, `no_conformidad_critica`, `no_conformidad_mayor`, `no_conformidad_menor`, `riesgo`.", type="csv")
    if uploaded_file:
        # Leer el archivo CSV en un DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Verificar si el CSV contiene las columnas necesarias
        required_columns = ["auditoria_id", "clausula", "no_conformidad_critica", "no_conformidad_mayor", "no_conformidad_menor", "riesgo"]
        if not all(column in df.columns for column in required_columns):
            st.error("El archivo CSV debe contener las columnas: " + ", ".join(required_columns))
        else:
            # Convertir la columna de riesgo a valores numéricos
            df["riesgo_numerico"] = df["riesgo"].map({"Alto": 2, "Medio": 1, "Bajo": 0})

            # Agrupar los datos por cláusula y calcular la suma de no conformidades
            grouped_df = df.groupby("clausula").agg({
                "no_conformidad_critica": "sum",
                "no_conformidad_mayor": "sum",
                "no_conformidad_menor": "sum",
                "riesgo_numerico": "first"  # Suponemos que el riesgo es el mismo para todas las auditorías de la cláusula
            }).reset_index()

            # Seleccionar características y objetivo
            X = grouped_df[["no_conformidad_critica", "no_conformidad_mayor", "no_conformidad_menor", "riesgo_numerico"]]
            y = df.groupby("clausula")["no_conformidad_critica"].apply(lambda x: 1 if x.sum() > 0 else 0)  # Evaluar si hay al menos una no conformidad crítica

            # Entrenar el modelo de árbol de decisiones
            model = DecisionTreeClassifier(random_state=42, max_depth=3)
            model.fit(X, y)

            # Visualizar el árbol de decisiones
            st.subheader("Visualización del Árbol de Decisiones")
            fig, ax = plt.subplots(figsize=(12, 8))
            plot_tree(model, feature_names=X.columns, class_names=["No Evaluar", "Evaluar"], filled=True, ax=ax)
            st.pyplot(fig)

            # Realizar predicciones
            grouped_df["evaluacion_predicha"] = model.predict(X)
            
            # Mostrar los resultados
            st.subheader("Resultados de Evaluación por Cláusula")
            st.write(grouped_df[["clausula", "no_conformidad_critica", "no_conformidad_mayor", "no_conformidad_menor", "riesgo", "evaluacion_predicha"]])

            # Descargar el modelo entrenado
            model_bytes = BytesIO()
            joblib.dump(model, model_bytes)
            model_bytes.seek(0)
            st.download_button("Descargar Modelo Entrenado", model_bytes, file_name="modelo_arbol_decisiones.pkl")

            # Descargar resultados como CSV
            result_csv = grouped_df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Resultados de Evaluación", result_csv, file_name="resultados_evaluacion.csv")
