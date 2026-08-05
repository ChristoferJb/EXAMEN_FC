import streamlit as st
import joblib
import re
import unicodedata
# 1. Configuración de la interfaz
st.set_page_config(page_title="Clasificador IA Educación", page_icon="🎓")
st.title("Análisis de Posturas: IA en la Universidad (Ecuador)")
st.markdown("**Grupo 07** - Minería de Datos. Clasificación de opiniones sobre el uso de Inteligencia Artificial.")

# 2. Cargar los artefactos matemáticos exportados
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load('modelo_clasificacion.pkl')
    vectorizador = joblib.load('vectorizador_tfidf.pkl')
    return modelo, vectorizador

modelo, vectorizador = cargar_modelos()

# 3. Replicar la funcion de limpieza (Pipeline estricto)
def limpiar_texto(texto):
    texto = str(texto).lower()
    
    # 1. CIRUGÍA DE MOJIBAKE: Reparar caracteres corruptos de Excel
    reparaciones = {
        "Ž": "e", "œ": "u", "‡": "a", "’": "i", "—": "o", "–": "n", "Ÿ": "u"
    }
    for mal, bien in reparaciones.items():
        texto = texto.replace(mal, bien)
        
    # 2. Quitar tildes reales (normalización)
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    
    # 3. Quitar ruido (URLs y símbolos)
    texto = re.sub(r"http\S+|www\.\S+", "", texto)
    texto = re.sub(r"[^\w\sñ]", "", texto) 
    
    # 4. Diccionario de jerga
    diccionario_jerga = {
        r"\bq\b": "que", r"\bxq\b": "porque", r"\bu\b": "universidad",
        r"\bia\b": "inteligencia artificial", r"\bchatgpt\b": "inteligencia artificial"
    }
    for p, r in diccionario_jerga.items():
        texto = re.sub(p, r, texto)
        
    return re.sub(r"\s+", " ", texto).strip()

# 4. Interfaz de Usuario
comentario = st.text_area("✍️ Ingresa el comentario de un estudiante ecuatoriano:", 
                          placeholder="Ej: Es trampa usar chatgpt, deberían prohibirlo en la U...")

if st.button("Analizar Postura"):
    if comentario.strip() == "":
        st.warning("Por favor, ingresa un comentario válido.")
    else:
        # Preprocesamiento
        texto_limpio = limpiar_texto(comentario)
        
        # Vectorización (TF-IDF)
        vector_tfidf = vectorizador.transform([texto_limpio])
        
        # Inferencia / Predicción
        prediccion = modelo.predict(vector_tfidf)[0]
        
        # Mapeo de salida para el usuario final
        if prediccion == 1:
            st.success("🟢 **Postura Detectada:** ACEPTADO (Apoya la integración de IA)")
        elif prediccion == 0:
            st.info("⚪ **Postura Detectada:** NEUTRAL (Pide reglas, normativas o es condicional)")
        else:
            st.error("🔴 **Postura Detectada:** RECHAZO (Condena el uso o pide prohibición)")
            
        st.caption(f"Texto procesado por el motor interno: *{texto_limpio}*")