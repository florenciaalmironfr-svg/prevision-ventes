# Pronóstico de ventas

Estructura base para desarrollar, entrenar y desplegar un modelo de machine learning para pronóstico de ventas.

## Inicio rápido

1. Crea y activa un entorno virtual de Python.
2. Instala las dependencias: `pip install -r requirements.txt`.
3. Ejecuta la aplicación: `streamlit run app.py`.

La app busca un archivo `data/processed/ventas.csv`. Si todavía no existe, muestra un estado vacío. Mantén los datos originales en `data/raw/` y guarda las transformaciones en `data/processed/`.

## Estructura

- `data/raw/`: datos originales, sin modificaciones.
- `data/interim/`: datos intermedios durante la preparación.
- `data/processed/`: datos limpios listos para análisis y entrenamiento.
- `notebooks/`: exploración, experimentos y prototipos.
- `src/data/`: carga y validación de datos.
- `src/features/`: limpieza y creación de variables.
- `src/models/`: entrenamiento y evaluación de modelos.
- `models/`: modelos entrenados y artefactos serializados.
- `app.py`: interfaz de usuario con Streamlit.
- `docs/`: documentación del proyecto.
- `reports/figures/`: gráficos exportados.
- `tests/`: pruebas automatizadas.

Los datos y modelos generados se excluyen de Git por defecto; los archivos `.gitkeep` conservan las carpetas vacías.
