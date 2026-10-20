from pathlib import Path

import math

import pandas as pd

from ml_fundamentos import (baseline_ingenuo, cargar_bike, detectar_fuga, metricas_regresion,
                            split_temporal, tabla_umbral)

DATOS = Path(__file__).resolve().parent.parent / "data" / "bike_day.csv"


# --- cargar_bike ---------------------------------------------------------------

def test_carga_los_731_dias_con_fecha_como_fecha():
    df = cargar_bike(DATOS)
    assert len(df) == 731
    assert pd.api.types.is_datetime64_any_dtype(df["fecha"])


def test_los_dias_quedan_en_orden_de_2011_a_2012():
    df = cargar_bike(DATOS)
    assert df["fecha"].iloc[0] == pd.Timestamp("2011-01-01")
    assert df["fecha"].iloc[-1] == pd.Timestamp("2012-12-31")
    assert df["fecha"].is_monotonic_increasing


def test_el_objetivo_conserva_su_valor_original():
    # fila 1 del CSV de UCI: 331 casuales + 654 registrados = 985 viajes
    df = cargar_bike(DATOS)
    assert df.loc[0, "viajes"] == 985


# --- split_temporal ------------------------------------------------------------

def test_split_temporal_deja_todo_2011_en_train_y_2012_en_test():
    df = cargar_bike(DATOS)
    train, test = split_temporal(df, "2012-01-01")
    assert len(train) == 365 and len(test) == 366       # 2012 fue bisiesto
    assert train["fecha"].max() < test["fecha"].min()


def test_split_temporal_no_pierde_ni_repite_filas():
    df = cargar_bike(DATOS)
    train, test = split_temporal(df, "2012-07-01")
    assert len(train) + len(test) == len(df)
    assert set(train["fecha"]).isdisjoint(test["fecha"])


# --- detectar_fuga -------------------------------------------------------------

def test_detecta_las_dos_columnas_que_suman_el_objetivo():
    df = cargar_bike(DATOS)
    assert set(detectar_fuga(df, "viajes")) == {"casuales", "registrados"}


def test_detecta_una_copia_escalada_del_objetivo():
    df = pd.DataFrame({"y": [10, 20, 35, 41], "y_en_miles": [0.010, 0.020, 0.035, 0.041],
                       "ruido": [3, 1, 4, 1]})
    assert detectar_fuga(df, "y") == ["y_en_miles"]


def test_no_marca_variables_legitimas():
    df = pd.DataFrame({"y": [10, 20, 35, 41], "precio": [5, 4, 9, 2], "ruido": [3, 1, 4, 1]})
    assert detectar_fuga(df, "y") == []


# --- baseline_ingenuo ----------------------------------------------------------

def test_baseline_predice_el_valor_de_hace_una_semana():
    y = pd.Series(range(1, 15))                      # 1, 2, ..., 14
    pred = baseline_ingenuo(y, rezago=7)
    assert pred.iloc[7] == 1 and pred.iloc[13] == 7
    assert pred.iloc[:7].isna().all()                # sin historia no hay predicción


# --- metricas_regresion --------------------------------------------------------

def test_metricas_regresion_con_un_ejemplo_resuelto_a_mano():
    m = metricas_regresion([3, 5, 7], [2, 5, 9])     # errores: -1, 0, 2
    assert m["MAE"] == 1.0
    assert math.isclose(m["RMSE"], math.sqrt(5 / 3))
    assert math.isclose(m["R2"], 0.375)              # 1 - 5/8


# --- tabla_umbral --------------------------------------------------------------

def test_tabla_umbral_cuenta_aciertos_y_errores_por_umbral():
    t = tabla_umbral([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], umbrales=[0.3, 0.5]).set_index("umbral")
    assert t.loc[0.5, ["VP", "FP", "FN", "VN"]].tolist() == [1, 1, 1, 1]
    assert t.loc[0.3, ["VP", "FP", "FN", "VN"]].tolist() == [2, 1, 0, 1]
    assert math.isclose(t.loc[0.3, "precision"], 2 / 3) and t.loc[0.3, "recall"] == 1.0
