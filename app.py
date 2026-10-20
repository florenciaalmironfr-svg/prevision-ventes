from pathlib import Path
import html
import pickle

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "modelo_final.joblib"
DATA_PATH = ROOT_DIR / "data" / "processed" / "inferencia_df_transformado.csv"
TRAINING_DATA_PATH = ROOT_DIR / "data" / "processed" / "df.csv"
LAG_COLUMNS = [f"unidades_vendidas_lag{i}" for i in range(1, 8)]
SCENARIOS = {
    "Actuel (0 %)": 1.0,
    "Concurrence -5 %": 0.95,
    "Concurrence +5 %": 1.05,
}
WEEKDAYS_FR = [
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
]
MODES_REMISE = [
    "Seulement le Black Friday",
    "Semaine du Black Friday",
    "Tout le mois",
    "Personnalisé",
]
JOURS_PAR_MODE = {
    "Seulement le Black Friday": [28],
    "Semaine du Black Friday": list(range(24, 31)),
    "Tout le mois": list(range(1, 31)),
}
BLACK_FRIDAY = pd.Timestamp("2025-11-28")


st.set_page_config(
    page_title="Simulateur de ventes · Novembre 2025",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f8ff 0%, #ffffff 42%);
    }
    h1, h2, h3 {
        color: #33265f;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border: 0;
        color: white;
        font-weight: 700;
        min-height: 3.2rem;
    }
    .metric-card {
        box-sizing: border-box;
        width: 100%;
        background: white;
        border: 1px solid #e8e8f5;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 14px rgba(70, 60, 130, 0.07);
        overflow-wrap: anywhere;
        word-wrap: break-word;
    }
    .metric-card-label {
        color: #5f6076;
        font-size: 0.92rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }
    .metric-card-value {
        color: #33265f;
        font-size: clamp(1.05rem, 2.3vw, 1.8rem);
        font-weight: 750;
        line-height: 1.2;
        overflow-wrap: anywhere;
        word-wrap: break-word;
        white-space: pre-line;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelo(ruta_modelo: str):
    return joblib.load(ruta_modelo)


@st.cache_data
def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    return pd.read_csv(ruta_csv, parse_dates=["fecha"])


@st.cache_data
def cargar_limites_historicos(ruta_csv: str) -> dict[str, float]:
    historico = pd.read_csv(
        ruta_csv,
        usecols=["descuento_porcentaje", "ratio_precio"],
    )
    return {
        "descuento_min": float(historico["descuento_porcentaje"].min()),
        "descuento_max": float(historico["descuento_porcentaje"].max()),
        "ratio_min": float(historico["ratio_precio"].min()),
        "ratio_max": float(historico["ratio_precio"].max()),
    }


def formato_euros(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", " ").replace(".", ",") + " €"


def formato_unidades(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", " ")


def formato_porcentaje(valor: float) -> str:
    return f"{valor:,.2f}".replace(",", " ").replace(".", ",") + " %"


def formato_porcentaje_signado(valor: float) -> str:
    signe = "−" if valor < 0 else "+" if valor > 0 else ""
    return f"{signe}{formato_porcentaje(abs(valor))}"


def afficher_carte(titre: str, valeur: str, couleur: str | None = None) -> None:
    style_couleur = f"color: {couleur};" if couleur else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{html.escape(titre)}</div>
            <div class="metric-card-value" style="{style_couleur}">
                {html.escape(valeur)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mettre_a_jour_jours_remise() -> None:
    mode = st.session_state["mode_remise"]
    if mode in JOURS_PAR_MODE:
        st.session_state["jours_remise"] = JOURS_PAR_MODE[mode].copy()


def simular_producto(
    modelo,
    producto_df: pd.DataFrame,
    nombres_features: list[str],
    ajuste_precio: int,
    jours_remise: list[int],
    factor_competencia: float,
) -> pd.DataFrame:
    """Prévoit novembre récursivement sans consulter d'autres données."""
    datos = producto_df.sort_values("fecha").reset_index(drop=True).copy()
    jours_remise_set = set(jours_remise)
    jours_avec_remise = datos["fecha"].dt.day.isin(jours_remise_set) & (
        ajuste_precio != 0
    )
    datos["precio_venta"] = datos["precio_base"].astype(float)
    datos.loc[jours_avec_remise, "precio_venta"] = (
        datos.loc[jours_avec_remise, "precio_base"] * (1 + ajuste_precio / 100)
    )
    datos["descuento_porcentaje"] = 0
    datos.loc[jours_avec_remise, "descuento_porcentaje"] = ajuste_precio
    datos["precio_competencia"] = datos["precio_competencia"] * factor_competencia

    if (datos["precio_competencia"] <= 0).any():
        raise ValueError("Le prix de la concurrence doit être supérieur à zéro.")

    datos["ratio_precio"] = datos["precio_venta"] / datos["precio_competencia"]
    historial_lags = [float(datos.iloc[0][columna]) for columna in LAG_COLUMNS]
    if not np.isfinite(historial_lags).all():
        raise ValueError(
            "Les retards du 1er novembre doivent être renseignés dans le CSV."
        )
    filas_resultado = []

    for posicion, fila in datos.iterrows():
        if posicion > 0:
            historial_lags = [prediccion_anterior, *historial_lags[:-1]]

        entrada = fila[nombres_features].to_frame().T.copy()
        for columna, valor in zip(LAG_COLUMNS, historial_lags):
            entrada.loc[entrada.index[0], columna] = valor
        entrada.loc[entrada.index[0], "unidades_vendidas_ma7"] = float(
            np.mean(historial_lags)
        )

        prediccion = max(0.0, float(modelo.predict(entrada)[0]))
        unidades_vendidas = max(0, int(np.floor(prediccion + 0.5)))
        ingreso = unidades_vendidas * float(fila["precio_venta"])
        filas_resultado.append(
            {
                "fecha": fila["fecha"],
                "jour_semaine": WEEKDAYS_FR[fila["fecha"].weekday()],
                "precio_base": float(fila["precio_base"]),
                "precio_venta": float(fila["precio_venta"]),
                "precio_competencia": float(fila["precio_competencia"]),
                "ratio_precio": float(fila["ratio_precio"]),
                "descuento_porcentaje": float(fila["descuento_porcentaje"]),
                "ajuste_aplicado": int(bool(jours_avec_remise.iloc[posicion])),
                "unidades_vendidas": unidades_vendidas,
                "ingresos_proyectados": ingreso,
            }
        )
        prediccion_anterior = prediccion

    return pd.DataFrame(filas_resultado)


if not MODEL_PATH.is_file():
    st.error(f"Modèle entraîné introuvable : `{MODEL_PATH}`")
    st.stop()

if not DATA_PATH.is_file():
    st.error(f"Fichier de données d'inférence introuvable : `{DATA_PATH}`")
    st.stop()

try:
    modelo = cargar_modelo(str(MODEL_PATH))
except (
    AttributeError,
    EOFError,
    ImportError,
    OSError,
    pickle.UnpicklingError,
    TypeError,
    ValueError,
) as error:
    st.error(f"Impossible de charger le modèle : {error}")
    st.stop()

try:
    datos = cargar_datos(str(DATA_PATH))
except (
    OSError,
    UnicodeDecodeError,
    ValueError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) as error:
    st.error(f"Impossible de charger les données d'inférence : {error}")
    st.stop()

if not TRAINING_DATA_PATH.is_file():
    st.error(f"Fichier d'entraînement introuvable : `{TRAINING_DATA_PATH}`")
    st.stop()

try:
    limites_historicos = cargar_limites_historicos(str(TRAINING_DATA_PATH))
except (
    OSError,
    ValueError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
) as error:
    st.error(f"Impossible de charger les plages historiques : {error}")
    st.stop()

if not hasattr(modelo, "feature_names_in_"):
    st.error("Le modèle ne fournit pas `feature_names_in_` ; l'ordre ne peut pas être vérifié.")
    st.stop()

feature_names = list(modelo.feature_names_in_)
columnas_app = [
    "fecha",
    "nombre",
    "precio_base",
    "precio_venta",
    "descuento_porcentaje",
    "precio_competencia",
    "ratio_precio",
    "unidades_vendidas_ma7",
    *LAG_COLUMNS,
]
faltan_columnas = sorted((set(feature_names) | set(columnas_app)) - set(datos.columns))
if faltan_columnas:
    st.error(f"Colonnes manquantes dans le CSV : {', '.join(faltan_columnas)}")
    st.stop()

fechas_noviembre = datos["fecha"].between(
    pd.Timestamp("2025-11-01"),
    pd.Timestamp("2025-11-30 23:59:59.999999"),
)
if not fechas_noviembre.all():
    st.error("Le CSV doit contenir uniquement des dates de novembre 2025.")
    st.stop()

productos = sorted(datos["nombre"].dropna().unique().tolist())
if not productos:
    st.error("Le CSV ne contient aucun produit à simuler.")
    st.stop()

st.title("📊 Simulation des ventes · Novembre 2025")

with st.sidebar:
    st.header("🎛️ Paramètres de simulation")
    producto_seleccionado = st.selectbox("Produit", productos)
    ajuste_precio = st.slider(
        "Ajustement du prix par rapport au prix de base",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        format="%d%%",
    )
    st.caption(
        "Plage observée dans l'historique : de "
        f"{formato_porcentaje_signado(limites_historicos['descuento_min'])} à "
        f"{formato_porcentaje_signado(limites_historicos['descuento_max'])}."
    )
    if "mode_remise" not in st.session_state:
        st.session_state["mode_remise"] = "Seulement le Black Friday"
    if "jours_remise" not in st.session_state:
        st.session_state["jours_remise"] = [28]
    st.radio(
        "Application de l'ajustement",
        MODES_REMISE,
        key="mode_remise",
        on_change=mettre_a_jour_jours_remise,
    )
    if st.session_state["mode_remise"] == "Personnalisé":
        st.multiselect(
            "Jours d'ajustement (novembre)",
            options=list(range(1, 31)),
            key="jours_remise",
        )
    escenario_seleccionado = st.radio(
        "Scénario de concurrence",
        list(SCENARIOS),
    )
    cout_unitaire = st.number_input(
        "Coût unitaire (optionnel, €)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        format="%.2f",
    )
    simular = st.button("🚀 Simuler les ventes", type="primary", width="stretch")

st.subheader(f"🛍️ Produit sélectionné : {producto_seleccionado}")
st.caption("Prévision quotidienne récursive pour les 30 jours de novembre 2025.")
jours_remise = sorted(st.session_state["jours_remise"])

firma_actual = (
    producto_seleccionado,
    ajuste_precio,
    tuple(jours_remise),
    escenario_seleccionado,
)

if simular:
    producto_df = datos.loc[datos["nombre"] == producto_seleccionado].copy()
    if len(producto_df) != 30 or producto_df["fecha"].nunique() != 30:
        st.error("Le produit sélectionné doit avoir exactement 30 dates distinctes.")
        st.stop()

    try:
        with st.spinner("Calcul de la prévision récursive et des scénarios..."):
            resultados_escenario = {
                nombre: simular_producto(
                    modelo,
                    producto_df,
                    feature_names,
                    ajuste_precio,
                    jours_remise,
                    factor,
                )
                for nombre, factor in SCENARIOS.items()
            }
            resultado_sin_ajuste = simular_producto(
                modelo,
                producto_df,
                feature_names,
                0,
                [],
                SCENARIOS[escenario_seleccionado],
            )
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"Impossible de terminer la simulation : {error}")
        st.stop()

    st.session_state["resultados_simulacion"] = resultados_escenario
    st.session_state["resultado_sin_ajuste"] = resultado_sin_ajuste
    st.session_state["firma_simulacion"] = firma_actual

resultados_guardados = st.session_state.get("resultados_simulacion")
if (
    resultados_guardados is None
    or st.session_state.get("firma_simulacion") != firma_actual
):
    st.info("Réglez les paramètres puis cliquez sur **Simuler les ventes** pour afficher la prévision.")
else:
    resultado = resultados_guardados[escenario_seleccionado]
    resultat_sans_ajustement = st.session_state["resultado_sin_ajuste"]
    resultado["hors_domaine_historique"] = (
        (resultado["descuento_porcentaje"] < limites_historicos["descuento_min"])
        | (resultado["descuento_porcentaje"] > limites_historicos["descuento_max"])
        | (resultado["ratio_precio"] < limites_historicos["ratio_min"])
        | (resultado["ratio_precio"] > limites_historicos["ratio_max"])
    )
    hors_domaine = bool(resultado["hors_domaine_historique"].any())
    if hors_domaine:
        pourcentage_prix_min = 100 + limites_historicos["descuento_min"]
        pourcentage_prix_max = 100 + limites_historicos["descuento_max"]
        st.warning(
            "⚠️ Scénario hors du domaine historique : dans les données "
            "2021-2024, le prix de vente a varié entre "
            f"{formato_porcentaje(pourcentage_prix_min)} et "
            f"{formato_porcentaje(pourcentage_prix_max)} du prix de base. "
            "Le modèle ne peut pas extrapoler au-delà : les résultats de ce "
            "scénario ne sont pas fiables."
        )
    jours_avec_ajustement = int(resultado["ajuste_aplicado"].sum())
    prix_apres_ajustement = resultado.loc[
        resultado["ajuste_aplicado"] == 1, "precio_venta"
    ]
    if prix_apres_ajustement.empty:
        prix_apres_ajustement = resultado["precio_venta"]
    signe_ajustement = "−" if ajuste_precio < 0 else "+" if ajuste_precio > 0 else ""
    valeur_ajustement = (
        f"{signe_ajustement}{abs(ajuste_precio)} % "
        f"({jours_avec_ajustement} "
        f"{'jour' if jours_avec_ajustement == 1 else 'jours'})"
    )

    st.divider()
    st.header("📌 Résumé de la prévision")
    kpi_gauche, kpi_droite = st.columns(2)
    with kpi_gauche:
        afficher_carte(
            "Unités totales",
            formato_unidades(resultado["unidades_vendidas"].sum()),
        )
        afficher_carte(
            "Prix de base",
            formato_euros(resultado["precio_base"].mean()),
        )
        afficher_carte(
            "Prix après ajustement",
            formato_euros(prix_apres_ajustement.mean()),
        )
    with kpi_droite:
        afficher_carte(
            "Chiffre d'affaires prévisionnel",
            formato_euros(resultado["ingresos_proyectados"].sum()),
        )
        afficher_carte(
            "Prix moyen de vente",
            formato_euros(resultado["precio_venta"].mean()),
        )
        afficher_carte(
            "Ajustement moyen",
            valeur_ajustement,
        )

    st.divider()
    st.header("📈 Unités prévues par jour")
    sns.set_theme(style="white")
    figura, eje = plt.subplots(figsize=(12, 4.5))
    sns.lineplot(
        data=resultado,
        x="fecha",
        y="unidades_vendidas",
        color="#667eea",
        linewidth=2.8,
        marker="o",
        markersize=5,
        ax=eje,
    )
    fila_black_friday = resultado.loc[resultado["fecha"] == BLACK_FRIDAY]
    if not fila_black_friday.empty:
        unidades_black_friday = float(
            fila_black_friday["unidades_vendidas"].iloc[0]
        )
        eje.axvline(
            BLACK_FRIDAY,
            color="#764ba2",
            linestyle="--",
            linewidth=1.5,
            alpha=0.85,
        )
        eje.scatter(
            [BLACK_FRIDAY],
            [unidades_black_friday],
            color="#e63946",
            s=90,
            zorder=5,
        )
        eje.annotate(
            "Black Friday",
            xy=(BLACK_FRIDAY, unidades_black_friday),
            xytext=(8, 15),
            textcoords="offset points",
            color="#a32335",
            fontweight="bold",
        )
    eje.set_xlabel("Jour de novembre")
    eje.set_ylabel("Unités prévues")
    eje.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    eje.xaxis.set_major_formatter(mdates.DateFormatter("%d nov."))
    leyenda = eje.get_legend()
    if leyenda is not None:
        leyenda.remove()
    figura.tight_layout()
    st.pyplot(figura, width="stretch")
    plt.close(figura)

    st.divider()
    st.header("🗓️ Détail quotidien")
    tabla = resultado.drop(columns=["hors_domaine_historique"]).copy()
    tabla["fecha"] = tabla["fecha"].map(
        lambda fecha: fecha.strftime("%d/%m/%Y")
        + (" 🛍️" if fecha == BLACK_FRIDAY else "")
    )
    tabla["fecha"] = tabla["fecha"] + np.where(
        resultado["hors_domaine_historique"],
        " ⚠️",
        "",
    )
    tabla = tabla.rename(
        columns={
            "fecha": "Date",
            "jour_semaine": "Jour de la semaine",
            "precio_base": "Prix de base",
            "precio_venta": "Prix après ajustement",
            "precio_competencia": "Prix de la concurrence",
            "descuento_porcentaje": "Ajustement",
            "ajuste_aplicado": "Ajustement appliqué",
            "unidades_vendidas": "Unités prévues",
            "ingresos_proyectados": "Chiffre d'affaires",
        }
    )
    tabla["Prix de base"] = tabla["Prix de base"].map(formato_euros)
    tabla["Prix après ajustement"] = tabla["Prix après ajustement"].map(formato_euros)
    tabla["Prix de la concurrence"] = tabla["Prix de la concurrence"].map(formato_euros)
    tabla["Ajustement"] = tabla["Ajustement"].map(formato_porcentaje_signado)
    tabla["Ajustement appliqué"] = tabla["Ajustement appliqué"].map(
        lambda valeur: "✅" if valeur and ajuste_precio != 0 else "—"
    )
    tabla["Unités prévues"] = tabla["Unités prévues"].map(formato_unidades)
    tabla["Chiffre d'affaires"] = tabla["Chiffre d'affaires"].map(formato_euros)

    def mettre_en_evidence_black_friday(fila):
        if "🛍️" in str(fila["Date"]):
            return ["background-color: #fff0f3; color: #7b2742; font-weight: bold"] * len(
                fila
            )
        return [""] * len(fila)

    tabla_estilizada = tabla.style.apply(mettre_en_evidence_black_friday, axis=1).hide(
        axis="index"
    )
    st.dataframe(
        tabla_estilizada,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="small"),
            "Jour de la semaine": st.column_config.TextColumn(
                "Jour de la semaine", width="small"
            ),
            "Prix de base": st.column_config.TextColumn(
                "Prix de base", width="small"
            ),
            "Prix après ajustement": st.column_config.TextColumn(
                "Prix après ajustement", width="small"
            ),
            "Prix de la concurrence": st.column_config.TextColumn(
                "Prix de la concurrence", width="small"
            ),
            "Ajustement": st.column_config.TextColumn("Ajustement", width="small"),
            "Ajustement appliqué": st.column_config.TextColumn(
                "Ajustement appliqué", width="small"
            ),
            "Unités prévues": st.column_config.TextColumn(
                "Unités prévues", width="small"
            ),
            "Chiffre d'affaires": st.column_config.TextColumn(
                "Chiffre d'affaires", width="small"
            ),
        },
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Les unités sont arrondies à l'entier le plus proche ; le chiffre "
        "d'affaires est calculé à partir des unités arrondies."
    )

    st.divider()
    st.header("💶 Impact économique de l'ajustement")
    chiffre_affaires_sans = float(
        resultat_sans_ajustement["ingresos_proyectados"].sum()
    )
    chiffre_affaires_avec = float(resultado["ingresos_proyectados"].sum())
    difference_chiffre_affaires = chiffre_affaires_avec - chiffre_affaires_sans
    difference_chiffre_affaires_pct = (
        difference_chiffre_affaires / chiffre_affaires_sans * 100
        if chiffre_affaires_sans
        else 0.0
    )
    ecart_volume = float(
        (
            (
                resultado["unidades_vendidas"]
                - resultat_sans_ajustement["unidades_vendidas"]
            )
            * resultado["precio_base"]
        ).sum()
    )
    ecart_prix = float(
        (
            resultado["unidades_vendidas"]
            * (resultado["precio_venta"] - resultado["precio_base"])
        ).sum()
    )
    ecart_controle = ecart_volume + ecart_prix - difference_chiffre_affaires
    controle_valide = abs(ecart_controle) <= 0.01
    unites_supplementaires = float(
        resultado["unidades_vendidas"].sum()
        - resultat_sans_ajustement["unidades_vendidas"].sum()
    )

    def couleur_valeur(valeur: float) -> str | None:
        if valeur > 0:
            return "#16803c"
        if valeur < 0:
            return "#c62828"
        return None

    st.subheader("💶 Chiffre d'affaires")
    cartes_ca = st.columns(3)
    with cartes_ca[0]:
        afficher_carte(
            "CA sans ajustement",
            formato_euros(chiffre_affaires_sans),
            couleur_valeur(chiffre_affaires_sans),
        )
    with cartes_ca[1]:
        afficher_carte(
            "CA avec ajustement",
            formato_euros(chiffre_affaires_avec),
            couleur_valeur(chiffre_affaires_avec),
        )
    with cartes_ca[2]:
        afficher_carte(
            "Écart de CA",
            f"{formato_euros(difference_chiffre_affaires)} "
            f"({formato_porcentaje(difference_chiffre_affaires_pct)})",
            couleur_valeur(difference_chiffre_affaires),
        )

    st.subheader("🔍 Analyse des écarts")
    cartes_ecarts = st.columns(3)
    with cartes_ecarts[0]:
        afficher_carte("Effet volume", formato_euros(ecart_volume), couleur_valeur(ecart_volume))
    with cartes_ecarts[1]:
        afficher_carte("Effet prix", formato_euros(ecart_prix), couleur_valeur(ecart_prix))
    with cartes_ecarts[2]:
        afficher_carte(
            "Contrôle",
            "Volume + prix = écart ✅"
            if controle_valide
            else f"Écart de contrôle : {formato_euros(ecart_controle)} ⚠️",
            "#16803c" if controle_valide else "#c62828",
        )

    st.subheader("📦 Volumes")
    cartes_volumes = st.columns(3)
    unites_sans_ajustement = int(resultat_sans_ajustement["unidades_vendidas"].sum())
    unites_avec_ajustement = int(resultado["unidades_vendidas"].sum())
    with cartes_volumes[0]:
        afficher_carte(
            "Unités sans ajustement",
            formato_unidades(unites_sans_ajustement),
            couleur_valeur(unites_sans_ajustement),
        )
    with cartes_volumes[1]:
        afficher_carte(
            "Unités avec ajustement",
            formato_unidades(unites_avec_ajustement),
            couleur_valeur(unites_avec_ajustement),
        )
    with cartes_volumes[2]:
        afficher_carte(
            "Unités supplémentaires",
            formato_unidades(unites_supplementaires),
            couleur_valeur(unites_supplementaires),
        )

    if difference_chiffre_affaires_pct > 100:
        st.warning(
            "Résultat atypique : un ajustement de prix appliqué sur une longue "
            "période peut être amplifié par la prévision récursive. Privilégiez "
            "une remise limitée à la période du Black Friday."
        )

    if cout_unitaire > 0:
        marge_sans = float(
            (
                (resultat_sans_ajustement["precio_venta"] - cout_unitaire)
                * resultat_sans_ajustement["unidades_vendidas"]
            ).sum()
        )
        marge_avec = float(
            (
                (resultado["precio_venta"] - cout_unitaire)
                * resultado["unidades_vendidas"]
            ).sum()
        )
        difference_marge = marge_avec - marge_sans
        difference_marge_pct = difference_marge / marge_sans * 100 if marge_sans else 0.0
        st.subheader("📈 Marge")
        cartes_marge = st.columns(3)
        with cartes_marge[0]:
            afficher_carte(
                "Marge sans ajustement",
                formato_euros(marge_sans),
                couleur_valeur(marge_sans),
            )
        with cartes_marge[1]:
            afficher_carte(
                "Marge avec ajustement",
                formato_euros(marge_avec),
                couleur_valeur(marge_avec),
            )
        with cartes_marge[2]:
            afficher_carte(
                "Écart de marge",
                f"{formato_euros(difference_marge)} "
                f"({formato_porcentaje(difference_marge_pct)})",
                couleur_valeur(difference_marge),
            )
    else:
        st.info(
            "Indiquez le coût unitaire pour estimer l'impact sur la marge : "
            "une hausse du chiffre d'affaires n'implique pas forcément une "
            "hausse du bénéfice."
        )

    st.subheader("📉 Décomposition du chiffre d'affaires")
    sns.set_theme(style="whitegrid")
    positions = np.arange(4)
    valeurs_waterfall = [
        chiffre_affaires_sans,
        ecart_volume,
        ecart_prix,
        chiffre_affaires_avec,
    ]
    bas_volume = chiffre_affaires_sans
    haut_volume = bas_volume + ecart_volume
    bas_prix = haut_volume
    haut_prix = bas_prix + ecart_prix
    bas_waterfall = [0, min(bas_volume, haut_volume), min(bas_prix, haut_prix), 0]
    hauteur_waterfall = [
        chiffre_affaires_sans,
        abs(ecart_volume),
        abs(ecart_prix),
        chiffre_affaires_avec,
    ]
    couleurs_waterfall = [
        "#667eea",
        "#16803c" if ecart_volume >= 0 else "#c62828",
        "#16803c" if ecart_prix >= 0 else "#c62828",
        "#667eea",
    ]
    figure_waterfall, axe_waterfall = plt.subplots(figsize=(12, 4.6))
    sns.barplot(
        x=positions,
        y=hauteur_waterfall,
        hue=positions,
        palette=couleurs_waterfall,
        legend=False,
        ax=axe_waterfall,
    )
    for position, base, hauteur, valeur in zip(
        positions, bas_waterfall, hauteur_waterfall, valeurs_waterfall
    ):
        rectangle = axe_waterfall.patches[position]
        rectangle.set_y(base)
        axe_waterfall.text(
            position,
            base + hauteur + max(chiffre_affaires_sans, chiffre_affaires_avec) * 0.015,
            formato_euros(valeur),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axe_waterfall.set_xticks(
        positions,
        ["CA sans ajustement", "Effet volume", "Effet prix", "CA avec ajustement"],
    )
    axe_waterfall.set_ylabel("Chiffre d'affaires (€)")
    axe_waterfall.set_xlabel("")
    axe_waterfall.set_ylim(
        0,
        max(chiffre_affaires_sans, chiffre_affaires_avec)
        + max(chiffre_affaires_sans, chiffre_affaires_avec) * 0.15,
    )
    figure_waterfall.tight_layout()
    st.pyplot(figure_waterfall, width="stretch")
    plt.close(figure_waterfall)

    st.divider()
    st.header("⚖️ Comparaison des scénarios de concurrence")
    tarjetas = st.columns(3)
    for tarjeta, (nombre_escenario, datos_escenario) in zip(
        tarjetas, resultados_guardados.items()
    ):
        with tarjeta:
            afficher_carte(
                f"{nombre_escenario} · Unités totales",
                formato_unidades(datos_escenario["unidades_vendidas"].sum()),
            )
            afficher_carte(
                f"{nombre_escenario} · Chiffre d'affaires",
                formato_euros(datos_escenario["ingresos_proyectados"].sum()),
            )

st.divider()
st.info(
    "Le modèle peut surestimer l'effet des remises appliquées en dehors du Black "
    "Friday, car dans les données historiques les fortes remises coïncidaient avec "
    "cet événement. Utilisez les simulations comme des scénarios indicatifs."
)
