from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "processed" / "ventas.csv"

st.set_page_config(page_title="Pronóstico de ventas", page_icon="📈", layout="wide")
st.title("Pronóstico de ventas")
st.caption("Explora los datos procesados del proyecto.")

if DATA_PATH.exists():
    sales = pd.read_csv(DATA_PATH)
    st.metric("Registros", f"{len(sales):,}")
    st.subheader("Vista previa")
    st.dataframe(sales.head(100), use_container_width=True)
else:
    st.info(
        "Todavía no hay datos procesados. Coloca el archivo "
        "`ventas.csv` en `data/processed/` para visualizarlo aquí."
    )
