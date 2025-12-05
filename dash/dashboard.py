# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import yaml

# -------------------------
# Utilities
# -------------------------
@st.cache_data
def load_config(path="config/pipeline_config.yaml"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

@st.cache_data
def load_data(path=None):
    cfg = load_config()
    default = cfg.get("paths", {}).get("processed_data", "data/processed/") + "merged_steam_twitch.csv"
    load_path = path or default
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No se encontró el archivo mergeado en: {load_path}")
    return pd.read_csv(load_path, low_memory=False)

def aggregate_by_game(df):
    df = df.copy()
    if "Game" in df.columns:
        df["Game"] = df["Game"].astype(str).str.strip().str.lower()
    elif "name" in df.columns:
        df["Game"] = df["name"].astype(str).str.strip().str.lower()

    numeric_cols = [
        c for c in ["Hours_watched", "Hours_streamed", "Peak_viewers",
                    "Peak_channels", "Streamers", "Avg_viewers",
                    "Avg_channels", "Avg_viewer_ratio"]
        if c in df.columns
    ]

    non_numeric = [c for c in df.columns if c not in numeric_cols]

    agg_dict = {c: "sum" for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric})

    return df.groupby("Game", as_index=False).agg(agg_dict)

# -------------------------
# Layout
# -------------------------
st.set_page_config(page_title="Steam + Twitch", layout="wide", initial_sidebar_state="expanded")

st.title("Correlación Steam + Twitch")
st.markdown(
    """
    Dashboard con datos combinados de **Steam** y **Twitch**.
    """
)

# -------------------------
# Cargar datos
# -------------------------
with st.spinner("Cargando datos..."):
    try:
        cfg = load_config()
        df_raw = load_data()
    except Exception as e:
        st.error(f"No se pudo cargar el archivo procesado: {e}")
        st.stop()

# -------------------------
# Preprocesamiento
# -------------------------
df = df_raw.copy()

#  Eliminar Enero 2016
if "Year" in df.columns and "Month" in df.columns:
    df = df[~((df["Year"] == 2016) & (df["Month"] == 1))]

for col in ["Game", "name"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

df_agg = aggregate_by_game(df)

# -------------------------
# Sidebar Filtros (sin género)
# -------------------------
st.sidebar.header("Filtros")

years = sorted(df_agg["Year"].dropna().unique().astype(int)) if "Year" in df_agg.columns else []
selected_year = st.sidebar.selectbox("Año", ["Todos"] + list(years), index=0)

top_n = st.sidebar.slider("Top N juegos", min_value=5, max_value=50, value=10)

is_free_filter = st.sidebar.selectbox("Mostrar Free/Paid", ["Ambos", "Free", "Paid"])

# Aplicación de filtros
dfv = df_agg.copy()

if selected_year != "Todos":
    dfv = dfv[dfv["Year"] == int(selected_year)]

if is_free_filter != "Ambos" and "is_free" in dfv.columns:
    dfv = dfv[dfv["is_free"] == (is_free_filter == "Free")]

st.sidebar.write(f"Registros filtrados: **{len(dfv):,}**")

# -------------------------
# KPIs
# -------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Juegos únicos", f"{df_agg.shape[0]:,}")
col2.metric("Horas vistas total", f"{int(df_agg['Hours_watched'].sum()):,}" if "Hours_watched" in df_agg else "N/A")
col3.metric("Streamers totales", f"{int(df_agg['Streamers'].sum()):,}" if "Streamers" in df_agg else "N/A")
col4.metric("Avg Viewers totales", f"{int(df_agg['Avg_viewers'].sum()):,}" if "Avg_viewers" in df_agg else "N/A")

# -------------------------
# PESTAÑAS
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Top Streamed",
    "Top Viewed",
    "Precio vs Viewers",
    "Correlación y Series",
    "Diccionario de Datos"
])

# =========================
# TAB 1: TOP STREAMED
# =========================
with tab1:
    st.subheader("Top juegos por horas transmitidas")

    if "Hours_streamed" in dfv.columns:
        top_streamed = dfv.sort_values("Hours_streamed", ascending=False).head(top_n)

        fig1 = px.bar(top_streamed, x="Game", y="Hours_streamed",
                      hover_data=["Streamers", "Avg_viewers"],
                      title=f"Top {top_n} - Horas transmitidas")
        st.plotly_chart(fig1, use_container_width=True)

        st.dataframe(top_streamed[["Game", "Hours_streamed", "Streamers", "Avg_viewers"]]
                     .reset_index(drop=True))

# =========================
# TAB 2: TOP VIEWED
# =========================
with tab2:
    st.subheader("Top juegos por horas vistas")

    if "Hours_watched" in dfv.columns:
        top_viewed = dfv.sort_values("Hours_watched", ascending=False).head(top_n)

        fig2 = px.bar(top_viewed, x="Game", y="Hours_watched",
                      hover_data=["Avg_viewers", "Peak_viewers"],
                      title=f"Top {top_n} - Horas vistas")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(top_viewed[["Game", "Hours_watched", "Avg_viewers", "Peak_viewers"]]
                     .reset_index(drop=True))

# =========================
# TAB 3: MODELOS PREDICTIVOS
# =========================
with tab3:
    st.subheader("Modelo Predictivo: Regresión Lineal")

    # ----------------------------------------
    # 1. Procesar precio sin mostrarlo
    # ----------------------------------------
    def extract_price_clean(x):
        if isinstance(x, dict) and "final" in x:
            return x["final"] / 100
        if isinstance(x, str):
            try:
                import ast
                obj = ast.literal_eval(x)
                if "final" in obj:
                    return obj["final"] / 100
            except:
                pass
        return np.nan

    dfv["price_numeric"] = dfv["price_overview"].apply(extract_price_clean)

    # =====================
    # USAR SOLO LOS JUEGOS TOP
    # =====================
    top_games = dfv.sort_values("Avg_viewers", ascending=False).head(30)
    model_df = top_games.dropna(subset=["price_numeric", "Avg_viewers"]).copy()

    # ============================================
    # 2. REGRESIÓN LINEAL: predecir Avg_viewers
    # ============================================
    import statsmodels.api as sm

    X = sm.add_constant(model_df["price_numeric"])
    y = model_df["Avg_viewers"]

    linear_model = sm.OLS(y, X).fit()

    # Mostrar solo métricas importantes
    st.markdown("###  Resumen del modelo")

    st.write(f"**R²:** {linear_model.rsquared:.4f}")
    st.write(f"**R² ajustado:** {linear_model.rsquared_adj:.4f}")
    st.write(f"**Coeficiente del precio:** {linear_model.params['price_numeric']:.4f}")
    st.write(f"**p-valor del coeficiente:** {linear_model.pvalues['price_numeric']:.4f}")

    # ========= EXPLICACIÓN EN EXPANDER DESPLEGABLE ==========
    with st.expander(" Explicación del modelo (haz clic para ver)"):
        st.markdown("""
### ¿Qué se hizo en este modelo?

1. **Se limpió el precio**  
   - Algunos precios vienen como diccionarios (`{"final": 999}`), por eso se extrae y transforma a dólares.
   - Se crea la variable `price_numeric`.

2. **Se seleccionaron solo los 30 juegos con más viewers.**  
   - Esto evita sesgos y da un dataset más homogéneo.

3. **Regresión Lineal para predecir Avg_viewers a partir del precio.**  
   - Fórmula:  
     \[
     \text{Avg\_viewers} = β_0 + β_1 \cdot \text{price\_numeric}
     \]

### ¿Qué significan las métricas?

- **R²**  
  Indica qué proporción de la variación de los viewers es explicada por el precio.  
  Valores cercanos a 1 = mejor modelo.

- **R² ajustado**  
  Igual que R² pero corrige cuando hay pocas variables.  
  Más honesto cuando el modelo es simple.

- **Coeficiente del precio (β₁)**  
  Te dice cuánto cambian los viewers cuando el precio aumenta 1 unidad.  
  - Si es negativo → juegos más caros tienden a tener menos viewers  
  - Si es positivo → juegos más caros tienden a tener más viewers

- **p-valor del coeficiente**  
  Indica si el efecto del precio es **estadísticamente significativo**.  
  - p < 0.05 → el precio realmente influye  
  - p > 0.05 → el precio NO tiene evidencia real de influir
        """)

    # ======================================================
    # 3. PREDICCIÓN DEL NÚMERO DE VIEWS HOY
    # ======================================================
    st.markdown("## Predicción: ¿Cuántas vistas tendría hoy un juego?")

    juego_pred = st.selectbox(
        "Selecciona un juego del TOP para predecir sus viewers:",
        model_df["Game"].unique()
    )

    fila = model_df[model_df["Game"] == juego_pred].iloc[0]
    precio_input = fila["price_numeric"]

    pred_viewers = linear_model.predict([1, precio_input])[0]

    st.success(f"Si **{juego_pred}** se jugara hoy, tendría aproximadamente **{pred_viewers:.0f} viewers**.")

# =========================
# TAB 4: CORRELACIÓN Y SERIES
# =========================
with tab4:
    st.subheader("Análisis de Correlación Interactiva")

    # ------------------------------------
    # 1. Seleccionar solo columnas numéricas
    # ------------------------------------
    numcols = [c for c in dfv.columns if dfv[c].dtype in [np.float64, np.int64]]

    if len(numcols) < 2:
        st.warning("No hay suficientes columnas numéricas para calcular correlación.")
        st.stop()

    # ------------------------------------
    # 2. Variables iniciales (primeras dos numéricas)
    # ------------------------------------
    default_vars = numcols[:2]

    # ------------------------------------
    # 3. Selección de variables adicionales
    # ------------------------------------
    st.markdown("### Selecciona las variables para analizar la correlación:")

    selected_vars = st.multiselect(
        "Variables disponibles:",
        numcols,
        default=default_vars
    )

    # Debe haber al menos 2 variables
    if len(selected_vars) < 2:
        st.warning("Selecciona mínimo dos variables para generar la matriz de correlación.")
        st.stop()

    # ------------------------------------
    # 4. Calcular la matriz de correlación
    # ------------------------------------
    corr = dfv[selected_vars].corr(method="pearson")

    # ------------------------------------
    # 5. Explicación de cómo se hace la correlación
    # ------------------------------------
    with st.expander("¿Cómo se calcula la correlación de Pearson?"):
        st.markdown("""
    La **correlación de Pearson** mide qué tan linealmente relacionadas están dos variables.
    
    Se calcula con la fórmula:

    \n\n
    **r = cov(X, Y) / (σₓ · σᵧ)**

    Donde:
    - **cov(X, Y)** = covarianza entre X y Y  
    - **σₓ** = desviación estándar de X  
    - **σᵧ** = desviación estándar de Y  

    Valores posibles:
    - **+1** = relación lineal totalmente positiva  
    - **0** = no existe relación lineal  
    - **–1** = relación lineal totalmente negativa  

    En esta matriz se calcula ese valor **para cada par de variables seleccionadas**.
    """)

    # ------------------------------------
    # 6. Mostrar la matriz
    # ------------------------------------
    st.markdown("##  Matriz de correlación con las variables seleccionadas")

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Matriz de correlación (Pearson)"
    )
    st.plotly_chart(fig_corr, use_container_width=True)


    st.subheader("Serie temporal - Horas vistas por mes")

    if all(c in df.columns for c in ["Year", "Month", "Hours_watched"]):
        time = df.groupby(["Year", "Month"], as_index=False)["Hours_watched"].sum()

        time["date"] = pd.to_datetime(
            time["Year"].astype(int).astype(str) + "-" +
            time["Month"].astype(int).astype(str) + "-01"
        )

        time = time.sort_values("date")

        fig_time = px.line(time, x="date", y="Hours_watched",
                           title="Horas vistas totales por mes")
        st.plotly_chart(fig_time, use_container_width=True)
# =========================
# TAB 5: DICCIONARIO DE DATOS
# =========================
with tab5:
    st.subheader("Diccionario de Datos del Dataset")

    st.write("A continuación se muestran todas las columnas del dataset junto con una definición corta y su tipo de dato.")

    # -------------------------------
    # Diccionario manual de definiciones
    # -------------------------------
    definiciones = {
        "Hours_watched": "Total de horas vistas del juego.",
        "Hours_streamed": "Total de horas transmitidas por streamers.",
        "Peak_viewers": "Número máximo de espectadores simultáneos.",
        "Peak_channels": "Máximo número de canales transmitiendo al mismo tiempo.",
        "Streamers": "Cantidad de streamers que transmitieron el juego.",
        "Avg_viewers": "Promedio de espectadores durante el periodo.",
        "Avg_channels": "Promedio de canales activos transmitiendo el juego.",
        "Avg_viewer_ratio": "Relación promedio entre viewers y canales.",
        "Rank": "Posición del juego en el ranking general.",
        "Year": "Año del registro.",
        "Month": "Mes del registro.",
        "type": "Tipo de contenido del juego.",
        "name": "Nombre del juego.",
        "steam_appid": "ID único del juego en Steam.",
        "required_age": "Edad mínima requerida para jugar.",
        "is_free": "Indica si el juego es gratuito.",
        "controller_support": "Tipo de soporte para control.",
        "dlc": "Contenido descargable disponible.",
        "detailed_description": "Descripción completa del juego.",
        "about_the_game": "Resumen detallado del juego.",
        "short_description": "Descripción corta del juego.",
        "fullgame": "Información relacionada a la edición completa.",
        "supported_languages": "Idiomas soportados por el juego.",
        "header_image": "Imagen principal del juego.",
        "website": "Sitio web oficial del juego.",
        "pc_requirements": "Requerimientos para PC.",
        "mac_requirements": "Requerimientos para Mac.",
        "linux_requirements": "Requerimientos para Linux.",
        "legal_notice": "Información legal del juego.",
        "drm_notice": "Aviso sobre DRM.",
        "ext_user_account_notice": "Requiere cuenta externa.",
        "developers": "Desarrolladores del juego.",
        "publishers": "Distribuidores o editores del juego.",
        "demos": "Demos disponibles del juego.",
        "price_overview": "Información detallada del precio.",
        "packages": "Paquetes disponibles del juego.",
        "package_groups": "Grupos de paquetes de Steam.",
        "platforms": "Plataformas en las que está disponible.",
        "metacritic": "Puntuación y reseñas de Metacritic.",
        "reviews": "Reseñas generales del juego.",
        "categories": "Categorías del juego.",
        "genres": "Géneros del juego.",
        "screenshots": "Imágenes del juego.",
        "movies": "Videos o trailers del juego.",
        "recommendations": "Número de recomendaciones.",
        "achievements": "Logros disponibles del juego.",
        "release_date": "Fecha de lanzamiento.",
        "support_info": "Información del soporte técnico.",
        "background": "Imagen o fondo del juego.",
        "content_descriptors": "Clasificación del contenido.",
        "Game": "Nombre del juego procesado.",
        "price_numeric": "Precio del juego convertido a número."
    }

    # -------------------------------
    # Construir tabla con nombres, tipos, definiciones
    # -------------------------------
    columnas = []
    tipos = []
    descs = []

    for col in dfv.columns:
        columnas.append(col)
        tipos.append(str(dfv[col].dtype))
        descs.append(definiciones.get(col, "Descripción no disponible."))

    dict_df = pd.DataFrame({
        "Columna": columnas,
        "Tipo de dato": tipos,
        "Descripción corta": descs
    })

    st.dataframe(dict_df, use_container_width=True)


# -------------------------
# Tabla final + descarga
# -------------------------
st.subheader("Tabla procesada (muestra)")
st.dataframe(dfv.head(200))

csv = dfv.to_csv(index=False).encode("utf-8")
st.download_button("Descargar CSV filtrado", csv, "filtered_merged.csv", "text/csv")
