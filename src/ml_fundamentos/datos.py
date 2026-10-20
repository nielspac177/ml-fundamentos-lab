"""Carga del dataset Bike Sharing (UCI) con nombres de columna en español."""
from pathlib import Path

import pandas as pd

NOMBRES = {
    "dteday": "fecha", "season": "temporada", "yr": "anio", "mnth": "mes",
    "holiday": "feriado", "weekday": "dia_semana", "workingday": "dia_laboral",
    "weathersit": "clima", "temp": "temperatura", "atemp": "sensacion_termica",
    "hum": "humedad", "windspeed": "viento", "casual": "casuales",
    "registered": "registrados", "cnt": "viajes",
}


def cargar_bike(ruta: str | Path) -> pd.DataFrame:
    """Lee bike_day.csv, renombra columnas y ordena por fecha. 'viajes' es el objetivo."""
    df = pd.read_csv(ruta, parse_dates=["dteday"]).drop(columns=["instant"])
    return df.rename(columns=NOMBRES).sort_values("fecha").reset_index(drop=True)
