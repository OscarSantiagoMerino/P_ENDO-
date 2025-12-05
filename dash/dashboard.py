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
    # carga el CSV procesado (por defecto desde config)
    cfg = load_config()
    default = cfg.get("paths", {}).get("processed_data", "data/processed/") + "merged_steam_twitch.csv"
    load_path = path or default
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No se encontró el archivo mergeado en: {load_path}")
    df = pd.read_csv(load_path, low_memory=False)
    return df

def safe_lower(x):
    try:
        return str(x).strip().lower()
    except:
        return x

def aggregate_by_game(df):
    # si ya está agregado, lo devolveremos agrupando igual para asegurar consistencia
    df = df.copy()
    # normalizar nombre de juego para agrupar
    if "Game" in df.columns:
        df["Game"] = df["Game"].astype(str).str.strip().str.lower()
    elif "name" in df.columns:
        df["Game"] = df["name"].astype(str).str.strip().str.lower()
    # columnas numéricas que queremos sumar (si existen)
    numeric_cols = [
        c for c in ["Hours_watched", "Hours_streamed", "Peak_viewers",
                    "Peak_channels", "Streamers", "Avg_viewers",
                    "Avg_channels", "Avg_viewer_ratio"]
        if c in df.columns
    ]
    # otras columnas que tomamos 'first'
    non_numeric = [c for c in df.columns if c not in numeric_cols]
    # construimos agg dict
    agg_dict = {c: "sum" for c in numeric_cols}
    agg_dict.update({c: "first" for c in non_numeric})
    aggregated = df.groupby("Game", as_index=False).agg(agg_dict)
    return aggregated

# -------------------------
# Layout
# -------------------------
st.set_page_config(page_title="Steam x Twitch Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title(" Dashboard: Correlación Steam ↔ Twitch")
st.markdown(
    """
    Este dashboard muestra métricas combinadas entre los datos de **Steam** y **Twitch**.
    Filtra por año/mes/género, explora top juegos, correlaciones y relaciones precio vs espectadores.
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
# asegurarnos de columnas en formato esperado
for col in ["Game", "name"]:
    if col in df.columns:
        df[col] = df[col].astype(str)

# agregamos por juego para evitar repeticiones
df_agg = aggregate_by_game(df)

# Sidebar - filtros
st.sidebar.header("Filtros")
years = sorted([int(x) for x in df_agg["Year"].dropna().unique()]) if "Year" in df_agg.columns else []
selected_year = st.sidebar.selectbox("Año", options=["Todos"] + years, index=0 if years else 0)
top_n = st.sidebar.slider("Top N juegos", min_value=5, max_value=50, value=10, step=1)
genre_filter = None
if "genres" in df_agg.columns:
    # intentar extraer género simple
    genres_series = df_agg["genres"].dropna().astype(str)
    # tomar parte textual si es lista-like '[{...}]'
    unique_genres = sorted(set(g.strip().lower() for val in genres_series for g in val.replace("'", "").replace("[", "").replace("]", "").split(",") if g.strip()))
    genre_filter = st.sidebar.selectbox("Filtrar por género (opcional)", options=["Todos"] + unique_genres, index=0)
is_free_filter = st.sidebar.selectbox("Mostrar sólo Free / Paid / Ambos", options=["Ambos", "Free", "Paid"])

# Aplicar filtros
dfv = df_agg.copy()
if selected_year != "Todos":
    if "Year" in dfv.columns:
        dfv = dfv[dfv["Year"] == int(selected_year)]
if genre_filter and genre_filter != "Todos" and "genres" in dfv.columns:
    # filtro básico: verificar substring
    dfv = dfv[dfv["genres"].astype(str).str.lower().str.contains(genre_filter)]
if is_free_filter != "Ambos":
    if "is_free" in dfv.columns:
        want_free = is_free_filter == "Free"
        dfv = dfv[dfv["is_free"] == want_free]

st.sidebar.markdown(f"Registros después del filtro: **{len(dfv):,}**")

# -------------------------
# KPIs (métricas generales)
# -------------------------
col1, col2, col3, col4 = st.columns(4)
total_games = df_agg.shape[0]
total_hours = int(df_agg["Hours_watched"].sum()) if "Hours_watched" in df_agg.columns else 0
total_streamers = int(df_agg["Streamers"].sum()) if "Streamers" in df_agg.columns else 0
avg_viewers = int(df_agg["Avg_viewers"].mean()) if "Avg_viewers" in df_agg.columns else 0

col1.metric("Juegos (únicos)", f"{total_games:,}")
col2.metric("Horas vistas (total)", f"{total_hours:,}")
col3.metric("Streamers (total)", f"{total_streamers:,}")
col4.metric("Avg viewers (med)", f"{avg_viewers:,}")

# -------------------------
# Top N juegos por Hours_streamed
# -------------------------
st.subheader("🎮 Top juegos por horas transmitidas")
if "Hours_streamed" in dfv.columns:
    top_streamed = dfv.sort_values("Hours_streamed", ascending=False).head(top_n)
    fig1 = px.bar(top_streamed, x="Game", y="Hours_streamed", hover_data=["Streamers", "Avg_viewers"], title=f"Top {top_n} - Horas transmitidas")
    st.plotly_chart(fig1, use_container_width=True)
    st.dataframe(top_streamed[["Game", "Hours_streamed", "Streamers", "Avg_viewers"]].reset_index(drop=True))
else:
    st.info("No existe la columna 'Hours_streamed' en el dataset.")

# -------------------------
# Top N juegos por Hours_watched
# -------------------------
st.subheader("👀 Top juegos por horas vistas (audiencia)")
if "Hours_watched" in dfv.columns:
    top_viewed = dfv.sort_values("Hours_watched", ascending=False).head(top_n)
    fig2 = px.bar(top_viewed, x="Game", y="Hours_watched", hover_data=["Avg_viewers", "Peak_viewers"], title=f"Top {top_n} - Horas vistas")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(top_viewed[["Game", "Hours_watched", "Avg_viewers", "Peak_viewers"]].reset_index(drop=True))
else:
    st.info("No existe la columna 'Hours_watched' en el dataset.")

# -------------------------
# Scatter: Precio vs Avg_viewers
# -------------------------
st.subheader("💶 Precio (Steam) vs Promedio de Viewers (Twitch)")
if "price_overview" in dfv.columns and "Avg_viewers" in dfv.columns:
    # intentamos extraer precio numeric (si price_overview es string que contiene 'final')
    def extract_price(x):
        try:
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                if "final" in x:
                    # extraer número entre 'final': 819  (simple)
                    import re
                    m = re.search(r"'final':\s*([0-9]+)", x)
                    if not m:
                        m = re.search(r'"final":\s*([0-9]+)', x)
                    if m:
                        return float(m.group(1)) / 100.0
                # si es string tipo '{"final":819}'
                import ast
                try:
                    obj = ast.literal_eval(x)
                    if isinstance(obj, dict) and "final" in obj:
                        return float(obj["final"]) / 100.0
                except Exception:
                    return np.nan
            if isinstance(x, dict):
                return float(x.get("final", np.nan)) / 100.0
        except Exception:
            return np.nan
    dfv["price_numeric"] = dfv["price_overview"].apply(extract_price)
    price_df = dfv.dropna(subset=["price_numeric", "Avg_viewers"])
    if not price_df.empty:
        fig3 = px.scatter(price_df, x="price_numeric", y="Avg_viewers", hover_data=["Game", "Hours_watched"], trendline="ols", title="Precio vs Avg_viewers")
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(price_df[["Game", "price_numeric", "Avg_viewers"]].sort_values("Avg_viewers", ascending=False).head(top_n).reset_index(drop=True))
    else:
        st.info("No hay datos de precio y avg_viewers juntos para mostrar.")
else:
    st.info("Falta 'price_overview' o 'Avg_viewers' para mostrar esta gráfica.")

# -------------------------
# Correlación
# -------------------------
st.subheader("🔗 Matriz de correlación (numérica)")
numcols = [c for c in dfv.columns if dfv[c].dtype in [np.float64, np.int64]]
if len(numcols) >= 2:
    corr = dfv[numcols].corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlación (Pearson) entre variables numéricas")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No hay suficientes columnas numéricas para calcular correlación.")

# -------------------------
# Series temporal (Horas vistas por mes)
# -------------------------
st.subheader("📈 Evolución temporal - Horas vistas por mes")
if all(c in df.columns for c in ["Year", "Month", "Hours_watched"]):
    time = df.groupby(["Year", "Month"], as_index=False)[["Hours_watched"]].sum()
    # crear fecha para ordenar
    time["date"] = pd.to_datetime(time["Year"].astype(int).astype(str) + "-" + time["Month"].astype(int).astype(str) + "-01")
    time = time.sort_values("date")
    fig_time = px.line(time, x="date", y="Hours_watched", title="Horas vistas totales (por mes)")
    st.plotly_chart(fig_time, use_container_width=True)
else:
    st.info("Faltan columnas Year/Month/Hours_watched para la serie temporal.")

# -------------------------
# Tabla completa y descarga
# -------------------------
st.subheader("📋 Tabla procesada (muestra)")
st.dataframe(dfv.head(200))

csv = dfv.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar CSV filtrado", csv, "filtered_merged.csv", "text/csv")

# -------------------------
# Explicación / Conclusiones
# -------------------------
st.sidebar.header("Sobre este dashboard")
st.sidebar.markdown(
    """
    - Los datos vienen de la unión entre Steam y Twitch (merge por nombre).
    - Se agrupan por `Game` sumando métricas numéricas para evitar repeticiones.
    - Filtra por año, género y free/paid.
    - El gráfico de Precio vs Avg_viewers intenta extraer precios de la columna `price_overview`.
    """
)

st.markdown("### 📝 Notas / recomendaciones")
st.markdown(
    """
    - Si observas juegos duplicados, revisa la limpieza de nombres (espacios, caracteres especiales).
    - Para dashboards más rápidos, guarda un archivo agregdo (`merged_steam_twitch_agg.csv`) desde el pipeline y úsalo aquí.
    - Para mayor detalle añade paginación o búsqueda por Game.
    """
)
