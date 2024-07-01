# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:54:50 2024

@author: jperezr
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize

# Generación de datos simulados
def generate_data():
    np.random.seed(42)
    refineries = ['Refinería A', 'Refinería B', 'Refinería C', 'Refinería D']
    data = {
        'Refinería': np.random.choice(refineries, 100),
        'Producción (barriles)': np.random.randint(1000, 5000, 100),
        'Eficiencia (%)': np.random.uniform(70, 100, 100),
        'Costos (USD)': np.random.uniform(10000, 50000, 100),
        'Emisiones CO2 (ton)': np.random.uniform(100, 500, 100)
    }
    return pd.DataFrame(data)

# Función para predecir producción
def predict_production(df, model_name):
    X = df[['Eficiencia (%)']]
    y = df['Producción (barriles)']
    
    if model_name == 'Linear Regression':
        model = LinearRegression()
    elif model_name == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_name == 'XGBoost':
        model = XGBRegressor(n_estimators=100, random_state=42)
    
    model.fit(X, y)
    df['Producción Predicha (barriles)'] = model.predict(X)
    return df, model

# Función para simulación de mantenimiento predictivo
def simulate_maintenance(df):
    df['Próximo Mantenimiento (días)'] = np.random.randint(30, 365, df.shape[0])
    return df

# Función de optimización de producción
def optimize_production(df):
    def objective(x):
        return -np.mean(x)

    constraints = [{'type': 'ineq', 'fun': lambda x: x - 70},
                   {'type': 'ineq', 'fun': lambda x: 100 - x}]
    
    result = minimize(objective, df['Eficiencia (%)'], constraints=constraints)
    df['Eficiencia Optimizada (%)'] = result.x
    return df

# Análisis de sensibilidad
def sensitivity_analysis(df, model):
    efficiency_range = np.linspace(70, 100, 100)
    predictions = model.predict(efficiency_range.reshape(-1, 1))
    return efficiency_range, predictions

# Crear aplicación en Streamlit
st.title('Análisis de Refinación en PEMEX')

# Mostrar datos simulados
df = generate_data()
st.subheader('Datos Simulados')
st.dataframe(df)

# Visualización de datos
st.subheader('Visualización de Producción')
fig = px.bar(df, x='Refinería', y='Producción (barriles)', title='Producción por Refinería')
st.plotly_chart(fig)

# Análisis de eficiencia
st.subheader('Análisis de Eficiencia')
fig = px.box(df, x='Refinería', y='Eficiencia (%)', title='Eficiencia por Refinería')
st.plotly_chart(fig)

# Análisis de costos
st.subheader('Análisis de Costos')
fig = px.bar(df, x='Refinería', y='Costos (USD)', title='Costos por Refinería')
st.plotly_chart(fig)

# Análisis de emisiones
st.subheader('Análisis de Emisiones de CO2')
fig = px.bar(df, x='Refinería', y='Emisiones CO2 (ton)', title='Emisiones de CO2 por Refinería')
st.plotly_chart(fig)

# Selección de modelo de machine learning
st.sidebar.subheader('Modelos de Machine Learning')
model_name = st.sidebar.selectbox('Selecciona un modelo', ('Linear Regression', 'Random Forest', 'XGBoost'))

# Predicción de producción
st.subheader('Predicción de Producción')
df, model = predict_production(df, model_name)
st.dataframe(df[['Refinería', 'Producción (barriles)', 'Producción Predicha (barriles)']])
mse = mean_squared_error(df['Producción (barriles)'], df['Producción Predicha (barriles)'])
r2 = r2_score(df['Producción (barriles)'], df['Producción Predicha (barriles)'])
st.write(f'Error Cuadrático Medio (MSE): {mse:.2f}')
st.write(f'Coeficiente de Determinación (R²): {r2:.2f}')

# Simulación de mantenimiento predictivo
st.subheader('Simulación de Mantenimiento Predictivo')
df = simulate_maintenance(df)
st.dataframe(df[['Refinería', 'Próximo Mantenimiento (días)']])

# Optimización de producción
st.subheader('Optimización de Producción')
df = optimize_production(df)
st.dataframe(df[['Refinería', 'Eficiencia (%)', 'Eficiencia Optimizada (%)']])

# Análisis de sensibilidad
st.subheader('Análisis de Sensibilidad')
efficiency_range, predictions = sensitivity_analysis(df, model)
fig = px.line(x=efficiency_range, y=predictions, labels={'x': 'Eficiencia (%)', 'y': 'Producción Predicha (barriles)'}, title='Análisis de Sensibilidad de la Producción')
st.plotly_chart(fig)

# Análisis de tendencias
st.subheader('Análisis de Tendencias')
selected_refineries = st.sidebar.multiselect('Selecciona las Refinerías para ver el análisis de tendencias', df['Refinería'].unique(), default=df['Refinería'].unique())
df_filtered = df[df['Refinería'].isin(selected_refineries)]

df_filtered['Fecha'] = pd.date_range(start='2023-01-01', periods=df_filtered.shape[0], freq='D')

fig = px.line(df_filtered, x='Fecha', y='Producción (barriles)', color='Refinería', title='Tendencia de Producción')
st.plotly_chart(fig)

fig = px.line(df_filtered, x='Fecha', y='Costos (USD)', color='Refinería', title='Tendencia de Costos')
st.plotly_chart(fig)

# Comparación internacional (simulada)
st.subheader('Comparación Internacional')
intl_data = {
    'País': ['México', 'EE.UU.', 'China', 'India', 'Rusia'],
    'Producción Promedio (barriles)': [3500, 7000, 6500, 3000, 4000],
    'Eficiencia Promedio (%)': [85, 90, 88, 75, 80]
}
intl_df = pd.DataFrame(intl_data)
st.dataframe(intl_df)

fig = px.bar(intl_df, x='País', y='Producción Promedio (barriles)', title='Producción Promedio por País')
st.plotly_chart(fig)

fig = px.bar(intl_df, x='País', y='Eficiencia Promedio (%)', title='Eficiencia Promedio por País')
st.plotly_chart(fig)

# Indicadores Clave de Desempeño (KPIs)
st.subheader('Indicadores Clave de Desempeño (KPIs)')
total_production = df['Producción (barriles)'].sum()
average_efficiency = df['Eficiencia (%)'].mean()
total_costs = df['Costos (USD)'].sum()
total_emissions = df['Emisiones CO2 (ton)'].sum()

st.write(f'Producción Total (barriles): {total_production}')
st.write(f'Eficiencia Promedio (%): {average_efficiency:.2f}')
st.write(f'Costos Totales (USD): {total_costs}')
st.write(f'Emisiones Totales CO2 (ton): {total_emissions}')

# Sección de ayuda
st.sidebar.title('Ayuda')
st.sidebar.subheader('Funcionalidades de la Aplicación')

st.sidebar.markdown("""
- **Importación de bibliotecas:** Importa las bibliotecas necesarias como Streamlit para la interfaz, Pandas y NumPy para manipulación de datos, Plotly Express para visualización, y varios modelos de regresión y métricas de evaluación de sklearn y scipy para análisis y predicción.

- **Generación de datos simulados:** Crea datos simulados que representan la producción, eficiencia, costos y emisiones de varias refinerías.

- **Funciones de predicción y modelado:**
  - **predict_production:** Utiliza modelos de regresión lineal, bosques aleatorios o XGBoost para predecir la producción de barriles en función de la eficiencia.
  - **simulate_maintenance:** Simula el próximo mantenimiento para cada refinería.
  - **optimize_production:** Optimiza la eficiencia de producción utilizando una función de optimización basada en la eficiencia actual.

- **Visualización de datos:** Utiliza Plotly Express para crear gráficos interactivos que muestran la producción, eficiencia, costos, emisiones y tendencias a lo largo del tiempo para las refinerías seleccionadas.

- **Selección de modelo y análisis de sensibilidad:** Permite al usuario seleccionar entre modelos de machine learning para la predicción de producción y realiza un análisis de sensibilidad sobre la eficiencia.

- **Comparación internacional:** Muestra una comparación simulada de la producción y eficiencia promedio por país.

- **Indicadores clave de desempeño (KPIs):** Calcula y muestra los KPIs como producción total, eficiencia promedio, costos totales y emisiones totales de CO2.

- **Sección de ayuda:** Proporciona información sobre cómo utilizar la aplicación y qué funcionalidades están disponibles.
""")