# src/orchestrator.py

from data_ingestion import ingest_data
from data_validation import validate_steam, validate_twitch
from data_transformation import merge_datasets, aggregate_by_game, save_processed


def run_pipeline():
    print("Ingestando datos...")
    steam_df, twitch_df, config = ingest_data()

    print("Validando datos...")
    validate_steam(steam_df)
    validate_twitch(twitch_df)

    print("Transformando y realizando merge...")
    merged = merge_datasets(steam_df, twitch_df)

    # 📌 Mostrar los años únicos ANTES de agrupar
    if "Year" in merged.columns:
        print("\n🟦 Años únicos detectados después del merge:")
        print(sorted(merged["Year"].unique()))
    else:
        print("\n⚠️ La columna 'Year' no existe en el dataframe mergeado.")

    merged = aggregate_by_game(merged)

    print("Guardando datos procesados...")
    output_file = save_processed(merged, config)

    print(f"\nPipeline completado! Archivo guardado en: {output_file}")


if __name__ == "__main__":
    run_pipeline()
