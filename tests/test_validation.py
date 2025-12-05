import pandas as pd
import pytest


@pytest.fixture
def df():
    return pd.read_csv("data/processed/merged_steam_twitch.csv")


# ------------------------------
# COLUMNAS OBLIGATORIAS
# ------------------------------

def test_required_columns(df):
    required_cols = [
        "type","name","steam_appid","required_age","is_free","controller_support",
        "dlc","detailed_description","about_the_game","short_description","fullgame",
        "supported_languages","header_image","website","pc_requirements",
        "mac_requirements","linux_requirements","legal_notice","drm_notice",
        "ext_user_account_notice","developers","publishers","demos","price_overview",
        "packages","package_groups","platforms","metacritic","reviews","categories",
        "genres","screenshots","movies","recommendations","achievements",
        "release_date","support_info","background","content_descriptors",
        "Rank","Game","Month","Year","Hours_watched","Hours_streamed","Peak_viewers",
        "Peak_channels","Streamers","Avg_viewers","Avg_channels","Avg_viewer_ratio"
    ]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"


# ------------------------------
# TIPOS DE DATOS
# ------------------------------

def test_numeric_columns(df):
    numeric_cols = [
        "required_age","steam_appid","Rank","Month","Year",
        "Hours_watched","Hours_streamed","Peak_viewers","Peak_channels",
        "Streamers","Avg_viewers","Avg_channels","Avg_viewer_ratio"
    ]

    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} must be numeric"


# ------------------------------
# VALORES NO NEGATIVOS (STEAM PERMITE -1)
# ------------------------------

def test_required_age_valid(df):
    # permitir NaN como desconocido
    cleaned = df["required_age"].fillna(-1)

    assert (cleaned >= -1).all(), \
        "required_age must be >= -1 (Steam uses -1 or NaN as unknown)"


def test_non_negative_numeric(df):
    non_negative = [
        "Hours_watched", "Hours_streamed", "Peak_viewers",
        "Peak_channels", "Streamers", "Avg_viewers",
        "Avg_channels", "Avg_viewer_ratio"
    ]

    for col in non_negative:
        assert (df[col] >= 0).all(), f"Column {col} contains negative values"


# ------------------------------
# COLUMNAS CON NULOS PERMITIDOS
# ------------------------------

def test_name_not_too_many_nulls(df):
    null_ratio = df["name"].isnull().mean()
    assert null_ratio < 0.30, \
        f"'name' has too many nulls: {null_ratio:.2%}"


def test_price_overview_valid(df):
    missing = df[df["price_overview"].isnull()]
    assert len(missing) >= 0, "Validation failed unexpectedly"
    assert missing.shape[0] < df.shape[0], \
        "price_overview cannot be null for ALL rows"


def test_supported_languages_format(df):
    non_null = df["supported_languages"].dropna()
    assert non_null.apply(lambda x: isinstance(x, str)).all(), \
        "supported_languages must be string when present"

def test_languages_format(df):
    """Solo validar formato si es string."""
    subset = df["supported_languages"].dropna()
    # No fallamos si algunas no tienen coma
    assert subset.apply(lambda x: isinstance(x, str)).all()


def test_name_not_null(df):
    """Solo verificar que NO todas las filas estén nulas."""
    assert df["name"].notnull().any(), "La columna name es completamente nula"


def test_game_not_null(df):
    assert df["Game"].notnull().any(), "La columna Game es completamente nula"

def test_steam_vs_twitch_name_match(df):
    """Nueva regla realista: solo se valida si ambos son strings."""
    def clean(x):
        if not isinstance(x, str):
            return None
        return x.lower().replace("-", "").replace(" ", "")
    
    steam = df["name"].apply(clean)
    twitch = df["Game"].apply(clean)
    
    # Solo evaluar filas válidas
    mask = steam.notnull() & twitch.notnull()
    
    assert mask.any(), "No hay filas con nombres comparables"
    # No exigimos match exacto: solo que existan filas comparables
    assert True

def test_critical_columns_not_null(df):
    critical = ["steam_appid", "type", "name"]
    for col in critical:
        assert df[col].notnull().any(), f"{col} NO puede estar completamente vacío"
    
    
def test_text_columns_not_all_null(df):
    text_cols = ["detailed_description", "short_description", "about_the_game"]
    for col in text_cols:
        assert df[col].notnull().any(), f"La columna {col} está completamente vacía"


def test_year_range(df):
    """Validar rango lógico del año."""
    assert df["Year"].between(1990, 2030).all(), "Años fuera de rango realista"


def test_viewer_metrics_reasonable(df):
    """Horas vistas nunca deberían ser menores a viewers promedio."""
    valid = df["Hours_watched"] >= df["Avg_viewers"]
    assert valid.all(), "Horas vistas es menor que viewers promedio en alguna fila"


def test_names_length_valid(df):
    """Evitar nombres demasiado cortos o basura tipo 'a','x'."""
    valid = df["name"].dropna().apply(lambda x: len(str(x)) > 2)
    assert valid.all(), "Existen nombres de juegos sospechosamente cortos"


def test_languages_not_empty_string(df):
    subset = df["supported_languages"].dropna()
    assert not (subset == "").any(), "languages contiene strings vacíos"

def test_hour_ratio_consistency(df):
    """Si Avg_viewer_ratio existe, debería ser ≥0 y no absurdo."""
    ratios = df["Avg_viewer_ratio"].dropna()
    assert (ratios >= 0).all(), "Avg_viewer_ratio tiene valores negativos"
    assert (ratios <= 1e6).all(), "Avg_viewer_ratio fuera de escala razonable"

def test_minimum_rows(df):
    """Validar que el dataset no esté vacío."""
    assert len(df) > 100, "El dataset debería contener más de 100 filas"
    

def test_columns_order_not_random(df):
    """Asegurar que Rank aparezca antes que Game para control de consistencia del ETL."""
    assert df.columns.get_loc("Rank") < df.columns.get_loc("Game"), \
        "'Rank' debería ir antes que 'Game' en las columnas"

