import streamlit as st
import numpy as np
import joblib
from scipy.stats import boxcox
import pandas as pd
from sklearn.svm import SVC

st.set_page_config(
    page_title="Trabajo Practico 1"
)

log_model = joblib.load("/modelos/log_model.pkl")
mlp_model = joblib.load("/modelos/mlp_model.pkl")
svm_model = joblib.load("/modelos/svm_model.pkl")

st.write("# Trabajo Practico n°1 👋")

st.markdown(
  """
  En esta pagina se muestran los distintos modelos desarrollados para el primer trabajo practico de la materia de **Inteligencia Artificial**.

  ## Integrantes
  - Facundo Bianchi
  - Delfina Perez
  - Martin Buzzetti
  - Nicolas Yague
  - Azour Barbar
  """
  )

st.title('Modelos')
st.markdown("""En esta página se puede probar los tres modelos 
  entrenados para predecir si un hongo es venenoso o comestible segun sus caracteristicas""")

st.subheader('Variables utilizadas')

st.markdown("""Para el modelo utilizamos como variables predictoras el 
**olor**, 
el **color de las laminas** y
el **color de las esporas**""")

odor={'Almendra': 0, 'Anis': 1, 'Creosota': 2, 'Pescado': 3, 'Feo': 4, 'Moho': 5, 'Ninguno': 6, 'Penetrante': 7, 'Picante': 8}
gillColor={'Negro': 0, 'Marron oscuro': 1, 'Ante / Marron claro': 2, 'Chocolate': 3, 'Gris': 4, 'Verde': 5, 'Naranja': 6, 'Rosa': 7, 'Violeta': 8, 'Rojo': 9, 'Blanco': 10, 'Amarillo': 11}
sporePrintColor={'Negro': 0, 'Marron oscuro': 1, 'Ante / Marron claro': 2, 'Chocolate': 3, 'Verde': 4, 'Naranja': 5, 'Violeta': 6, 'Blanco': 7, 'Amarillo': 8}

col1, col2, col3 = st.columns(3)

with col1:
  st.write(f'<h4 class="big-font"> Olor </h4>', unsafe_allow_html=True)
  st.write("")
  st.write("")
  olor = st.selectbox('Seleccione una opción', options=list(odor.keys()))
  olorn = odor[olor]

with col2:
  st.write(f'<h4 class="big-font"> Color de las laminas </h4>', unsafe_allow_html=True)
  cLaminas = st.selectbox('Seleccione una opción',options=list(gillColor.keys()))
  cLaminasn= gillColor[cLaminas]

with col3:
  st.write(f'<h4 class="big-font"> Color de las Esporas </h4>', unsafe_allow_html=True)
  cEsporas = col3.selectbox('Seleccione una opción', options=list(sporePrintColor.keys()))
  cEsporasn = sporePrintColor[cEsporas]

variables={
      "odor": olorn,
      "gill-color": cLaminasn,
      "spore-print-color": cEsporasn}

variables = pd.DataFrame([variables])

col1, col2, col3 = st.columns(3)

with col1:
  st.markdown("### Modelo de Regresion Logistica(LR)")
  st.markdown(f'La prediccion del modelo es que el hongo es **<span style="color:red;">{"Comestible" if log_model.predict(variables) == 0 else "Venenoso"}</span>**', unsafe_allow_html=True)
  st.markdown("Este modelo predice con un 74% de presicion")

with col2:
  st.markdown("### Modelo de Red Neuronal(MLP)")
  st.markdown(f'La prediccion del modelo es que el hongo es **<span style="color:red;">{"Comestible" if mlp_model.predict(variables) == 0 else "Venenoso"}</span>**', unsafe_allow_html=True)
  st.markdown("Este modelo predice con un 97% de presicion")

with col3:
  st.markdown("### Modelos de Maquina de Soporte Vectorial(SVM)")
  st.markdown(f'La prediccion del modelo es que el hongo es **<span style="color:red;">{"Comestible" if svm_model.predict(variables) == 0 else "Venenoso"}</span>**', unsafe_allow_html=True)
  st.markdown("Este modelo predice con un 98% de presicion")

