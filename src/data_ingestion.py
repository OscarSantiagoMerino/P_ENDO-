# src/data_ingestion.py
import pandas as pd
import yaml
import os

def load_config(path="config/pipeline_config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_safely(path, dtypes=None):
    """
    Carga CSV de forma robusta:
    - intenta UTF-8
    - si falla usa Latin-1
    - aplica dtype seguro
    - rellena NA con 'NA'
    - elimina filas totalmente vacías
    """
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=dtypes)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", dtype=dtypes)

    # elimina filas sin datos
    df = df.dropna(how="all")

    # rellena NA sin desplazar columnas
    df = df.fillna("NA")

    return df


def ingest_data():
    config = load_config()

    raw_path = config["paths"]["raw_data"]
    steam_file = os.path.join(raw_path, config["files"]["steam_dataset"])
    twitch_file = os.path.join(raw_path, config["files"]["twitch_dataset"])

    print(f"Cargando Steam desde: {steam_file}")

    steam_dtypes = {
        "name": "string",
        "steam_appid": "string",
        "price_overview": "string",     
        "developers": "string",
        "publishers": "string"
    }

    steam_df = load_csv_safely(steam_file, dtypes=steam_dtypes)


    print(f"Cargando Twitch desde: {twitch_file}")

    twitch_dtypes = {
        "Game": "string",
        "Month": "Int64",   # se carga como entero seguro
        "Year": "Int64",
        "Hours_watched": "Int64",
        "Hours_streamed": "Int64",
        "Peak_viewers": "Int64",
        "Peak_channels": "Int64",
        "Streamers": "Int64",
        "Avg_viewers": "float",
        "Avg_channels": "float",
        "Avg_viewer_ratio": "float",
        "Rank": "Int64"
    }

    twitch_df = load_csv_safely(twitch_file, dtypes=twitch_dtypes)

    return steam_df, twitch_df, config
