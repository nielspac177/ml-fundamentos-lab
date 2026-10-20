"""Partición temporal, detección de fuga y métricas para el laboratorio."""
import pandas as pd


def split_temporal(df: pd.DataFrame, fecha_corte: str, columna: str = "fecha"):
    """Train = antes de fecha_corte; test = desde fecha_corte. Nunca mezcla futuro con pasado."""
    corte = pd.Timestamp(fecha_corte)
    return df[df[columna] < corte].copy(), df[df[columna] >= corte].copy()


def detectar_fuga(df: pd.DataFrame, objetivo: str, umbral: float = 0.99) -> list[str]:
    """Columnas sospechosas de fuga: casi idénticas al objetivo (|r| >= umbral)
    o que, sumadas de a dos, reproducen el objetivo fila por fila."""
    numericas = [c for c in df.select_dtypes("number").columns if c != objetivo]
    y = df[objetivo]
    sospechosas = {c for c in numericas if abs(df[c].corr(y)) >= umbral}
    for i, a in enumerate(numericas):
        for b in numericas[i + 1:]:
            if ((df[a] + df[b]) - y).abs().max() < 1e-9:
                sospechosas |= {a, b}
    return [c for c in numericas if c in sospechosas]


def baseline_ingenuo(y: pd.Series, rezago: int = 7) -> pd.Series:
    """Predice cada día con el valor de hace `rezago` días. Cualquier modelo debe ganarle."""
    return y.shift(rezago)


def metricas_regresion(y_real, y_pred) -> dict:
    """MAE, RMSE y R² sin dependencias extra, para poder leer cada fórmula."""
    real, pred = pd.Series(y_real, dtype=float), pd.Series(y_pred, dtype=float)
    error = real.values - pred.values
    sse = float((error ** 2).sum())
    sst = float(((real - real.mean()) ** 2).sum())
    return {"MAE": float(abs(error).mean()), "RMSE": (sse / len(error)) ** 0.5, "R2": 1 - sse / sst}


def tabla_umbral(y_real, score, umbrales) -> pd.DataFrame:
    """Matriz de confusión, precision y recall para cada umbral de decisión."""
    y = pd.Series(y_real).astype(int).values
    s = pd.Series(score, dtype=float).values
    filas = []
    for u in umbrales:
        pred = (s >= u).astype(int)
        vp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        fn, vn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        filas.append({"umbral": u, "VP": vp, "FP": fp, "FN": fn, "VN": vn,
                      "precision": vp / (vp + fp) if vp + fp else float("nan"),
                      "recall": vp / (vp + fn) if vp + fn else float("nan")})
    return pd.DataFrame(filas)
