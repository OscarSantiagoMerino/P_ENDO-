import pandas as pd
import os
import yaml
import numpy as np


def load_config():
    with open("config/pipeline_config.yaml", "r") as f:
        return yaml.safe_load(f)


def load_merged(path="data/processed/merged_steam_twitch.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError("El archivo merge no existe. Corre ingest + transform + validation.")
    return pd.read_csv(path)


# ============================================================
# ====================== ANÁLISIS BÁSICO ======================
# ============================================================

def basic_stats(df):
    numeric = df.select_dtypes(include=[np.number])
    return numeric.describe()


def compute_correlations(df):
    """Solo Spearman, se quitó Pearson completamente"""
    numeric = df.select_dtypes(include=[np.number])
    corr_spearman = numeric.corr(method="spearman")
    return corr_spearman


def top_viewed_games(df, top_n=10):
    return df.sort_values("Hours_watched", ascending=False).head(top_n)[
        ["Game", "Hours_watched", "Avg_viewers"]
    ]


# ============================================================
# =================== ANÁLISIS ADICIONALES ====================
# ============================================================

def price_vs_viewers(df):
    """Relación entre precio y viewers"""
    if "price_overview" in df.columns:
        df["price_numeric"] = df["price_overview"].apply(
            lambda x: eval(x).get("final", None)/100 if isinstance(x, str) and "final" in x else None
        )

    price_view = df[["Game", "price_numeric", "Avg_viewers"]].dropna()
    return price_view.sort_values("Avg_viewers", ascending=False).head(15)


def games_by_genre(df):
    """Conteo de juegos por género."""
    if "genres" not in df.columns:
        return pd.DataFrame()

    genre_counts = (
        df["genres"]
        .fillna("[]")
        .apply(lambda x: [g.strip() for g in x.strip("[]").replace("'", "").split(",")] if isinstance(x, str) else [])
    )

    exploded = pd.DataFrame({"genre": sum(genre_counts.tolist(), [])})
    return exploded["genre"].value_counts().head(20)


def most_streamed_games(df, top_n=10):
    return df.sort_values("Hours_streamed", ascending=False).head(top_n)[
        ["Game", "Hours_streamed", "Streamers"]
    ]


def viewer_efficiency(df):
    if "Avg_viewer_ratio" not in df.columns:
        return pd.DataFrame()

    return df.sort_values("Avg_viewer_ratio", ascending=False).head(15)[
        ["Game", "Avg_viewer_ratio", "Avg_viewers", "Avg_channels"]
    ]


# ============================================================
# =================== GUARDAR REPORTE =========================
# ============================================================

def save_report(
    stats, corr_s, top_games,
    price_rel, genre_count, top_streamed, efficiency
):
    output_path = "data/processed/analysis_report.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("==== ESTADÍSTICAS BÁSICAS ====\n")
        f.write(stats.to_string())

        f.write("\n\n==== CORRELACIÓN SPEARMAN ====\n")
        f.write(corr_s.to_string())

        f.write("\n\n==== TOP 10 JUEGOS MÁS VISTOS ====\n")
        f.write(top_games.to_string())

        f.write("\n\n==== RELACIÓN PRECIO ↔ VIEWERS ====\n")
        f.write(price_rel.to_string())

        f.write("\n\n==== JUEGOS POR GÉNERO ====\n")
        f.write(genre_count.to_string())

        f.write("\n\n==== JUEGOS MÁS STREAMEADOS ====\n")
        f.write(top_streamed.to_string())

        f.write("\n\n==== EFICIENCIA DE VIEWERS (Avg_viewer_ratio) ====\n")
        f.write(efficiency.to_string())

    print(f"Reporte guardado en: {output_path}")


# ============================================================
# ====================== EJECUCIÓN ============================
# ============================================================

def run_analysis():
    config = load_config()
    df = load_merged()

    print("Análisis iniciado...\n")

    stats = basic_stats(df)
    corr_s = compute_correlations(df)
    top_games = top_viewed_games(df)
    price_rel = price_vs_viewers(df)
    genre_count = games_by_genre(df)
    top_streamed = most_streamed_games(df)
    efficiency = viewer_efficiency(df)

    save_report(
        stats, corr_s, top_games,
        price_rel, genre_count,
        top_streamed, efficiency
    )

    print("\nAnálisis completado.")


if __name__ == "__main__":
    run_analysis()
