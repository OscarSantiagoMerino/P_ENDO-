# src/data_validation.py

def validate_steam(df):
    required_cols = [
        "name", "steam_appid", "type", "price_overview",
        "developers", "publishers"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna {col} en steam_app_data.csv")
    return True


def validate_twitch(df):
    required_cols = [
        "Game", "Year", "Month", "Hours_watched",
        "Avg_viewers", "Rank"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna {col} en Twitch_game_data.csv")
    return True
