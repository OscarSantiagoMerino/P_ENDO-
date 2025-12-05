# Proyecto Steam — Pipeline de Análisis Steam + Twitch

## 1. Objetivos

Este proyecto tiene como objetivo analizar cómo se relacionan métricas provenientes de videojuegos publicados en Steam (precio, género, calificación y popularidad) con métricas sociales y de audiencia provenientes de Twitch (horas vistas, espectadores promedio y ranking de popularidad).

Los objetivos específicos son:

- Construir un pipeline automatizado para procesar datos de ambas plataformas.
- Integrar diversas fuentes de datos y validarlas.
- Identificar relaciones significativas entre el rendimiento comercial y el rendimiento social.
- Aplicar principios DataOps para lograr reproducibilidad, monitoreo y escalabilidad.
- Generar resultados automáticos y trazables.

---

## 2. Diseño y Arquitectura

El proyecto está organizado como un pipeline modular:

```
Proyecto_Steam/
├─ config/
│  └─ pipeline_config.yaml
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ data_ingestion.py
│  ├─ data_transformation.py
│  ├─ data_validation.py
│  ├─ analysis.py
│  └─ orchestrator.py
├─ tests/
└─ requirements.txt
```

Los componentes principales del pipeline son:

### data_ingestion.py

- Carga los datos desde CSV utilizando rutas configurables.
- Soporta datos locales o remotos.
- Manejo de archivos grandes mediante Git LFS.

### data_transformation.py

- Limpieza de nulos y duplicados.
- Estandarización de tipos de datos.
- Unión de tablas Steam + Twitch.

### data_validation.py

- Verifica la calidad del dataset.
- Asegura que no existan columnas inválidas o inconsistentes.
- Reglas de validación centralizadas.

### analysis.py

- Correlaciones entre variables (Spearman).
- Estadísticos descriptivos.
- Comparación entre los juegos con mayor popularidad.

### orchestrator.py

- Ejecuta todo el pipeline end‑to‑end.
- Produce una versión final limpia de los datos.
- Registra logs para auditoría.

---

## 3. Metodología

La metodología aplicada fue la siguiente:

1. Extracción de datos desde Steam y Twitch.
2. Preprocesamiento: normalización, conversión de tipos, limpieza.
3. Validación de datos mediante reglas automáticas.
4. Transformación y fusión de datasets.
5. Análisis estadístico:
   - Correlación Spearman.
   - Análisis de categorías y precios.
6. Registro de resultados y logs.
7. Ejecución automática mediante CI/CD.

---

## 4. Resultados clave

Los resultados se generan automáticamente en `data/processed/`.

Algunos hallazgos esperados:

- Los juegos con precios menores tienden a tener mayor audiencia.
- Ciertos géneros son más consumidos en Twitch independientemente del precio.
- Las métricas sociales pueden predecir popularidad en Steam.

Los resultados cambian conforme los datos se actualizan en el pipeline.

---

## 5. Pruebas automatizadas y logs

Los tests se ejecutan con:

```powershell
pytest -q
```

Se realizaron pruebas unitarias para:

- Validación de datos.
- Transformaciones.
- Correlaciones.
- Integración del pipeline completo.

### Logs

El pipeline produce logs automáticos sobre:

- Ingesta de datos.
- Transformaciones.
- Errores durante ejecución.
- Estadísticas del procesamiento.

Estos logs son accesibles tanto en ejecución local como en CI.

---

## 6. Reflexión sobre principios DataOps aplicados

Se aplicaron los siguientes principios DataOps:

- Automatización del flujo de datos de inicio a fin.
- Control de versiones y trazabilidad de datos mediante Git LFS.
- Pipeline reproducible tanto local como en CI/CD.
- Validaciones automáticas para garantizar calidad del dato.
- Módulos desacoplados para mantenimiento incremental.
- Observabilidad mediante logs y pruebas automáticas.

El proyecto asegura que cualquier persona pueda ejecutar el pipeline y obtener los mismos resultados sin intervención manual.

---

## 7. Ejecutar el pipeline

```powershell
python src/orchestrator.py
```

Esto generará:

- Datos transformados.
- Reportes de análisis.
- Resultados estadísticos.

---

## 8. CI/CD (GitHub Actions)

Archivo ubicado en:

```
.github/workflows/ci.yaml
```

Se ejecuta automáticamente para:

- Instalar dependencias.
- Cargar datos desde Git LFS.
- Ejecutar pipeline y tests.
- Publicar artefactos como logs y resultados.

---

## 9. Requisitos

- Python 3.10+
- Paquetes de requirements.txt
- Git LFS para archivos grandes

---

## 10. Contacto

Proyecto mantenido por **OscarSantiagoMerino**.
