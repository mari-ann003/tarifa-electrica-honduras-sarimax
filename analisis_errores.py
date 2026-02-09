import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
import itertools
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG (WINDOWS)
# =========================
BASE_DIR = Path(r"C:\Users\marian.padilla\predictor_tarf")

PATH_CBG = BASE_DIR / "cree_costo_generacion_quarterly_2021_2026_from_reports.csv"
PATH_TAR = BASE_DIR / "tarifa_residencial_m_50_trimestral.csv"
PATH_TC  = BASE_DIR / "tc_trimestral_referencia.csv"

TEST_SIZE = 4  # últimos 4 trimestres como test
H = 4          # horizonte de forecast (trimestres)

OUT_METRICS      = BASE_DIR / "model_metrics_con_diagnostico_v2.csv"
OUT_CV           = BASE_DIR / "cv_results_v2.csv"
OUT_SENS         = BASE_DIR / "sens_results_v2.csv"
OUT_DIAGNOSTIC   = BASE_DIR / "diagnostico_modelo_v2.csv"

# =========================
# HELPERS
# =========================
def period_to_index(p: str) -> int:
    year = int(p[:4])
    q = int(p[-1])
    return year * 4 + q

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    eps = 1e-9
    return np.mean(np.abs(y_true - y_pred) / np.maximum(denom, eps)) * 100

def metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_%": float(mape(y_true, y_pred)),
        "sMAPE_%": float(smape(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }

def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

# =========================
# 1) LOAD DATA
# =========================
print_header("1) CARGANDO DATOS")

df_cbg = pd.read_csv(PATH_CBG)
df_tar = pd.read_csv(PATH_TAR)
df_tc  = pd.read_csv(PATH_TC)

required_tar_cols = {"periodo", "energia_hnl_kwh", "cargo_fijo_hnl_mes"}
required_cbg_cols = {"periodo", "costo_generacion_usd_mwh"}
required_tc_cols  = {"periodo", "tc_trimestre_promedio"}

missing_tar = required_tar_cols - set(df_tar.columns)
missing_cbg = required_cbg_cols - set(df_cbg.columns)
missing_tc  = required_tc_cols  - set(df_tc.columns)

if missing_tar:
    raise ValueError(f"Faltan columnas en tarifa CSV: {missing_tar}")
if missing_cbg:
    raise ValueError(f"Faltan columnas en CBG CSV: {missing_cbg}")
if missing_tc:
    raise ValueError(f"Faltan columnas en TC CSV: {missing_tc}")

# =========================
# 2) BUILD MODEL TABLE
# =========================
print_header("2) CONSTRUYENDO TABLA DE MODELO")

df = (
    df_tar[["periodo", "energia_hnl_kwh", "cargo_fijo_hnl_mes"]]
    .merge(df_cbg[["periodo", "costo_generacion_usd_mwh"]], on="periodo", how="left")
    .merge(df_tc[["periodo", "tc_trimestre_promedio"]], on="periodo", how="left")
)

df["cbg_hnl_kwh"] = (df["costo_generacion_usd_mwh"] * df["tc_trimestre_promedio"]) / 1000.0
df["period_index"] = df["periodo"].map(period_to_index)
df = df.sort_values("period_index").reset_index(drop=True)

df["tarifa_lag1"] = df["energia_hnl_kwh"].shift(1)

model_df = df.dropna(subset=["energia_hnl_kwh", "cbg_hnl_kwh", "tc_trimestre_promedio"]).copy()
model_df_ols = model_df.dropna(subset=["tarifa_lag1"]).copy()

print(f"Períodos totales disponibles: {len(model_df)}")
print(f"Rango temporal: {model_df['periodo'].iloc[0]} a {model_df['periodo'].iloc[-1]}")

if len(model_df) <= TEST_SIZE + 8:
    TEST_SIZE = max(3, len(model_df)//4)
    print(f"Ajustando TEST_SIZE a {TEST_SIZE} por tamaño de muestra")

train_sar = model_df.iloc[:-TEST_SIZE].copy()
test_sar  = model_df.iloc[-TEST_SIZE:].copy()

train_ols = model_df_ols.iloc[:-TEST_SIZE].copy()
test_ols  = model_df_ols.iloc[-TEST_SIZE:].copy()

print(f"Train períodos: {len(train_sar)} ({train_sar['periodo'].iloc[0]} a {train_sar['periodo'].iloc[-1]})")
print(f"Test períodos: {len(test_sar)} ({test_sar['periodo'].iloc[0]} a {test_sar['periodo'].iloc[-1]})")

exog_cols = ["cbg_hnl_kwh", "tc_trimestre_promedio"]

# =========================
# 3) DIAGNÓSTICO INICIAL
# =========================
print_header("3) DIAGNÓSTICO INICIAL: ESTADÍSTICAS Y CORRELACIONES")

desc_stats = pd.DataFrame({
    "Variable": ["Tarifa (HNL/kWh)", "CBG (HNL/kWh)", "TC (HNL/USD)"],
    "Media": [
        model_df["energia_hnl_kwh"].mean(),
        model_df["cbg_hnl_kwh"].mean(),
        model_df["tc_trimestre_promedio"].mean()
    ],
    "Std": [
        model_df["energia_hnl_kwh"].std(),
        model_df["cbg_hnl_kwh"].std(),
        model_df["tc_trimestre_promedio"].std()
    ],
    "Min": [
        model_df["energia_hnl_kwh"].min(),
        model_df["cbg_hnl_kwh"].min(),
        model_df["tc_trimestre_promedio"].min()
    ],
    "Max": [
        model_df["energia_hnl_kwh"].max(),
        model_df["cbg_hnl_kwh"].max(),
        model_df["tc_trimestre_promedio"].max()
    ],
    "CV%": [
        model_df["energia_hnl_kwh"].std() / model_df["energia_hnl_kwh"].mean() * 100,
        model_df["cbg_hnl_kwh"].std() / model_df["cbg_hnl_kwh"].mean() * 100,
        model_df["tc_trimestre_promedio"].std() / model_df["tc_trimestre_promedio"].mean() * 100
    ]
})
print(desc_stats.to_string(index=False))

print("\nMatriz de correlaciones:")
corr_matrix = model_df[["energia_hnl_kwh", "cbg_hnl_kwh", "tc_trimestre_promedio"]].corr()
print(corr_matrix.round(3))

# =========================
# 4) BENCHMARKS NAÏVE
# =========================
print_header("4) PRUEBA 1: BENCHMARKS NAÏVE")

naive_last = np.array([train_sar["energia_hnl_kwh"].iloc[-1]] * len(test_sar))
m_last = metrics(test_sar["energia_hnl_kwh"], naive_last)

naive_mean = np.array([train_sar["energia_hnl_kwh"].mean()] * len(test_sar))
m_mean = metrics(test_sar["energia_hnl_kwh"], naive_mean)

if len(train_sar) >= 2:
    last_val = train_sar["energia_hnl_kwh"].iloc[-1]
    first_val = train_sar["energia_hnl_kwh"].iloc[0]
    drift_per_period = (last_val - first_val) / (len(train_sar) - 1)
    naive_drift = np.array([last_val + drift_per_period * (i+1) for i in range(len(test_sar))])
    m_drift = metrics(test_sar["energia_hnl_kwh"], naive_drift)
else:
    m_drift = {"R2": np.nan, "MAPE_%": np.nan, "RMSE": np.nan}

bench_df = pd.DataFrame([
    {"Modelo": "Naïve (último valor)", **m_last},
    {"Modelo": "Naïve (promedio histórico)", **m_mean},
    {"Modelo": "Naïve (drift/tendencia)", **m_drift},
])[["Modelo","R2","MAPE_%","RMSE"]]
print(bench_df.to_string(index=False))

# =========================
# 5) ENTRENAR SARIMAX + TEST
# =========================
print_header("5) ENTRENANDO SARIMAX Y EVALUANDO EN TEST")

sarimax = SARIMAX(
    train_sar["energia_hnl_kwh"],
    exog=train_sar[exog_cols],
    order=(1, 0, 0),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sar_fore = sarimax.get_forecast(steps=len(test_sar), exog=test_sar[exog_cols])
sar_pred = pd.Series(sar_fore.predicted_mean.values, index=test_sar.index, name="pred_sarimax")

m_sar = metrics(test_sar["energia_hnl_kwh"], sar_pred)

print(f"AIC: {sarimax.aic:.2f} | BIC: {sarimax.bic:.2f} | LLF: {sarimax.llf:.2f}")
print("Métricas SARIMAX (test):", m_sar)

bench_plus = pd.concat([
    pd.DataFrame([{"Modelo": "SARIMAX(1,0,0)+exog", "R2": m_sar["R2"], "MAPE_%": m_sar["MAPE_%"], "RMSE": m_sar["RMSE"]}]),
    bench_df
], ignore_index=True)
print("\nComparativa actualizada:")
print(bench_plus.to_string(index=False))

# =========================
# 6) CV TEMPORAL (TimeSeriesSplit)
# =========================
print_header("6) PRUEBA 2: VALIDACIÓN CRUZADA TEMPORAL")

n_splits = 4
# Nota: con 20 entradas, 4 splits suele funcionar; si falla, hay que bajar la cantidad de n_splits.
cv_results = []
for try_splits in [4, 3, 2]:
    tscv = TimeSeriesSplit(n_splits=try_splits)
    cv_results = []
    ok = True
    for fold, (train_idx, val_idx) in enumerate(tscv.split(model_df), 1):
        train_cv = model_df.iloc[train_idx]
        val_cv = model_df.iloc[val_idx]
        if len(train_cv) < 8 or len(val_cv) < 2:
            continue
        try:
            model_cv = SARIMAX(
                train_cv["energia_hnl_kwh"],
                exog=train_cv[exog_cols],
                order=(1, 0, 0),
                trend="c",
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            pred_cv = model_cv.get_forecast(steps=len(val_cv), exog=val_cv[exog_cols]).predicted_mean
            m = metrics(val_cv["energia_hnl_kwh"], pred_cv)

            cv_results.append({
                "fold": fold,
                "train_range": f"{train_cv['periodo'].iloc[0]}-{train_cv['periodo'].iloc[-1]}",
                "test_range": f"{val_cv['periodo'].iloc[0]}-{val_cv['periodo'].iloc[-1]}",
                "R2": m["R2"],
                "MAPE_%": m["MAPE_%"],
                "RMSE": m["RMSE"]
            })
        except Exception:
            ok = False
            break
    if ok and len(cv_results) >= 2:
        break

cv_df = pd.DataFrame(cv_results)
if len(cv_df) == 0:
    print("No se pudo ejecutar CV temporal de forma estable con estos datos.")
else:
    print(cv_df.to_string(index=False))
    print("\nResumen CV:")
    print(f"R2 promedio: {cv_df['R2'].mean():.4f} | std: {cv_df['R2'].std():.4f}")
    print(f"MAPE promedio: {cv_df['MAPE_%'].mean():.2f}% | RMSE promedio: {cv_df['RMSE'].mean():.4f}")

# =========================
# 7) RESIDUOS: one-step ahead en train, Ljung-Box, Shapiro y Durbin-Watson
# =========================
print_header("7) PRUEBA 3: ANÁLISIS DE RESIDUOS (TRAIN one-step ahead)")

# One-step ahead en TRAIN (desde el segundo dato, porque hay AR(1))
start_ix = train_sar.index[1]
end_ix   = train_sar.index[-1]

pred_in = sarimax.get_prediction(
    start=start_ix,
    end=end_ix,
    exog=train_sar.loc[start_ix:end_ix, exog_cols]
).predicted_mean

y_in = train_sar.loc[start_ix:end_ix, "energia_hnl_kwh"]
res_train = (y_in.values - pred_in.values)

print(f"n residuos train: {len(res_train)}")
print(f"Media: {res_train.mean():.6f} | Std: {res_train.std():.6f} | Min: {res_train.min():.6f} | Max: {res_train.max():.6f}")

# Shapiro-Wilk (indicativo)
try:
    sw_stat, sw_p = shapiro(res_train)
    print(f"Shapiro-Wilk p-value (indicativo): {sw_p:.6f}")
except Exception as e:
    sw_p = np.nan
    print("Shapiro falló:", str(e))

# Ljung-Box: usar lags pequeños, no mayores que n/5 aprox
lags = [1, 2, 4]
lags = [l for l in lags if l < len(res_train)]
try:
    lb = acorr_ljungbox(res_train, lags=lags, return_df=True)
    print("\nLjung-Box (H0: no autocorrelación):")
    print(lb.round(6))
except Exception as e:
    lb = None
    print("Ljung-Box falló:", str(e))

# Durbin-Watson
dw = sm.stats.stattools.durbin_watson(res_train)
print(f"\nDurbin-Watson: {dw:.6f}  (≈2 es deseable)")

# =========================
# 8) SENSIBILIDAD A TAMAÑO DE ENTRENAMIENTO
# =========================
print_header("8) PRUEBA 4: SENSIBILIDAD A TAMAÑO DE ENTRENAMIENTO")

train_sizes = [8, 12, len(train_sar)]
sens_results = []

for size in train_sizes:
    if size < 8 or size >= len(model_df) - 2:
        continue

    train_sens = model_df.iloc[-(size+TEST_SIZE):-TEST_SIZE].copy()
    test_sens  = model_df.iloc[-TEST_SIZE:].copy()

    try:
        model_sens = SARIMAX(
            train_sens["energia_hnl_kwh"],
            exog=train_sens[exog_cols],
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        pred_sens = model_sens.get_forecast(steps=len(test_sens), exog=test_sens[exog_cols]).predicted_mean
        m = metrics(test_sens["energia_hnl_kwh"], pred_sens)

        sens_results.append({
            "train_size": size,
            "train_range": f"{train_sens['periodo'].iloc[0]}-{train_sens['periodo'].iloc[-1]}",
            "R2": m["R2"],
            "MAPE_%": m["MAPE_%"],
            "RMSE": m["RMSE"]
        })
    except Exception:
        continue

sens_df = pd.DataFrame(sens_results)
if len(sens_df) == 0:
    print("No se pudo ejecutar sensibilidad.")
else:
    print(sens_df.to_string(index=False))
    r2_range = sens_df["R2"].max() - sens_df["R2"].min()
    print(f"\nRango de R2 entre tamaños: {r2_range:.6f} (más pequeño = más estable)")

# =========================
# 9) SHUFFLE TEST (n=4 -> 24 permutaciones)
# =========================
print_header("9) PRUEBA 5: SHUFFLE TEST EXACTO (p-valor empírico)")

y_true = test_sar["energia_hnl_kwh"].values
y_pred = sar_pred.values
r2_obs = r2_score(y_true, y_pred)

r2_perm = []
for perm in itertools.permutations(y_true):
    r2_perm.append(r2_score(np.array(perm), y_pred))
r2_perm = np.array(r2_perm)

p_emp = (1 + np.sum(r2_perm >= r2_obs)) / (1 + len(r2_perm))

print(f"R2 observado: {r2_obs:.6f}")
print(f"R2 permutado (mean): {r2_perm.mean():.6f}")
print(f"R2 permutado (max):  {r2_perm.max():.6f}")
print(f"p-valor empírico exacto: {p_emp:.6f}")
print("Interpretación: p pequeño (<0.05) sugiere que el R2 observado es raro bajo permutación.")

# =========================
# 10) COMPARACIÓN CON OLS
# =========================
print_header("10) COMPARACIÓN CON OLS (baseline)")

X_train = train_ols[["cbg_hnl_kwh", "tc_trimestre_promedio", "tarifa_lag1"]]
X_train = sm.add_constant(X_train)
y_train = train_ols["energia_hnl_kwh"]

ols = sm.OLS(y_train, X_train).fit()

X_test = test_ols[["cbg_hnl_kwh", "tc_trimestre_promedio", "tarifa_lag1"]]
X_test = sm.add_constant(X_test)
ols_pred = ols.predict(X_test)

m_ols = metrics(test_ols["energia_hnl_kwh"], ols_pred)

final_compare = pd.DataFrame([
    {"Modelo": "SARIMAX(1,0,0)+exog", "R2": m_sar["R2"], "MAPE_%": m_sar["MAPE_%"], "RMSE": m_sar["RMSE"]},
    {"Modelo": "OLS + lag1 + exog",   "R2": m_ols["R2"], "MAPE_%": m_ols["MAPE_%"], "RMSE": m_ols["RMSE"]},
    {"Modelo": "Naïve (último valor)","R2": m_last["R2"],"MAPE_%": m_last["MAPE_%"],"RMSE": m_last["RMSE"]},
])
print(final_compare.to_string(index=False))

# =========================
# 11) EXPORTAR CSVs DE SOPORTE
# =========================
print_header("11) EXPORTANDO CSVs")

# Métricas principales
metrics_df = pd.DataFrame([
    {"Modelo": "SARIMAX(1,0,0)+exog", **m_sar, "AIC": sarimax.aic, "BIC": sarimax.bic, "LLF": sarimax.llf},
    {"Modelo": "OLS + lag1 + exog",   **m_ols}
])
metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8")

# CV y sensibilidad
if len(cv_df) > 0:
    cv_df.to_csv(OUT_CV, index=False, encoding="utf-8")
if len(sens_df) > 0:
    sens_df.to_csv(OUT_SENS, index=False, encoding="utf-8")

# Diagnóstico resumido
diag_rows = [
    {"test": "Test principal", "valor": m_sar["R2"], "detalle": "R2 en test (n=4)"},
    {"test": "Test principal", "valor": m_sar["MAPE_%"], "detalle": "MAPE% en test (n=4)"},
    {"test": "Benchmark", "valor": m_last["MAPE_%"], "detalle": "MAPE% naïve último valor"},
    {"test": "Benchmark", "valor": m_mean["MAPE_%"], "detalle": "MAPE% naïve promedio"},
    {"test": "Shuffle exacto", "valor": p_emp, "detalle": "p-valor empírico exacto (24 permutaciones)"},
    {"test": "Residuos train", "valor": dw, "detalle": "Durbin-Watson (≈2 deseable)"},
]
if not np.isnan(sw_p):
    diag_rows.append({"test": "Residuos train", "valor": sw_p, "detalle": "Shapiro p-value (indicativo)"})

diagnostic_df = pd.DataFrame(diag_rows)
diagnostic_df.to_csv(OUT_DIAGNOSTIC, index=False, encoding="utf-8")

print("Archivos generados:")
print(" -", OUT_METRICS)
print(" -", OUT_DIAGNOSTIC)
if len(cv_df) > 0:   print(" -", OUT_CV)
if len(sens_df) > 0: print(" -", OUT_SENS)

