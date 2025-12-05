# src/data_transformation.py

import pandas as pd
import numpy as np

# ------------------------------------------------------
# LIMPIEZA DE STEAM
# ------------------------------------------------------

def clean_steam(df):
    df = df.copy()

    # Normalizar nombre del juego
    df["name"] = df["name"].astype(str).str.strip().str.lower()

    return df


# ------------------------------------------------------
# LIMPIEZA AVANZADA DE TWITCH
# ------------------------------------------------------

def clean_year_column(df):
    df = df.copy()

    # Convertir a número
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

    # INVALIDAR años fuera de rango razonable
    df.loc[(df["Year"] < 2012) | (df["Year"] > 2025), "Year"] = np.nan

    # IMPUTACIÓN POR MODO POR CADA JUEGO
    df["Year"] = df.groupby("Game")["Year"].transform(
        lambda x: x.fillna(x.mode().iloc[0]) if not x.mode().empty else x
    )

    # IMPUTAR LOS NaN RESTANTES POR EL MODO GLOBAL
    if df["Year"].isna().sum() > 0:
        global_mode = df["Year"].mode().iloc[0]
        df["Year"].fillna(global_mode, inplace=True)

    # A ENTERO FINAL
    df["Year"] = df["Year"].astype(int)

    return df


def clean_twitch(df):
    df = df.copy()

    # Normalizar nombre del juego
    df["Game"] = df["Game"].astype(str).str.strip().str.lower()

    # Limpiar y corregir Año
    df = clean_year_column(df)

    return df


# ------------------------------------------------------
# MERGE STEAM + TWITCH
# ------------------------------------------------------

def merge_datasets(steam_df, twitch_df):
    steam_df = clean_steam(steam_df)
    twitch_df = clean_twitch(twitch_df)

    merged = steam_df.merge(
        twitch_df,
        left_on="name",
        right_on="Game",
        how="inner"
    )

    return merged


# ------------------------------------------------------
# AGRUPACIÓN FINAL PARA EVITAR DUPLICADOS
# ------------------------------------------------------

def aggregate_by_game(df):

    if "Game" not in df.columns:
        raise ValueError("La columna 'Game' no existe en el dataframe mergeado.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Quitar columnas que NO deben sumarse
    for col in ["Year", "Month"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    aggregated = df.groupby("Game", as_index=False).agg(
        # Sumamos solo métricas numéricas
        {col: "sum" for col in numeric_cols} |

        # Year → modo del año
        {"Year": lambda x: int(x.mode()[0]) if not x.mode().empty else int(x.dropna().iloc[0])} |

        # Month → modo también
        {"Month": lambda x: int(x.mode()[0]) if not x.mode().empty else int(x.dropna().iloc[0])} |

        # El resto → primer valor
        {col: "first" for col in non_numeric_cols}
    )

    return aggregated



# ------------------------------------------------------
# GUARDADO FINAL
# ------------------------------------------------------

def save_processed(df, config):
    output_path = config["paths"]["processed_data"] + "merged_steam_twitch.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path
