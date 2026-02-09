import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG (WINDOWS)
# =========================
BASE_DIR = Path(r"C:\Users\marian.padilla\predictor_tarf")

PATH_CBG = BASE_DIR / "cree_costo_generacion_quarterly_2021_2026_from_reports.csv"
PATH_TAR = BASE_DIR / "alumbrado_publico_2021_2025.csv"                 
PATH_TC  = BASE_DIR / "tc_trimestral_referencia.csv"

TEST_SIZE = 4  # últimos 4 trimestres como test
H = 4         # horizonte de forecast (trimestres), cambiar según se desee

OUT_METRICS  = BASE_DIR / "model_metrics_alumbrado_publico.csv"          
OUT_PREDS    = BASE_DIR / "model_predictions_alumbrado_publico.csv"     
OUT_FORECAST = BASE_DIR / "forecast_alumbrado_publico_escenarios.csv"    

# CSV extra (el que me pediste que se nombre así)
OUT_CF_INFL = BASE_DIR / "cargo_fijo_alumbrado_publico_2026_inflacion.csv"  

# =========================
# HELPERS
# =========================
def period_to_index(p: str) -> int:
    year = int(p[:4])
    q = int(p[-1])
    return year * 4 + q

def index_to_period(idx: int) -> str:
    year = idx // 4
    q = idx % 4
    if q == 0:
        year -= 1
        q = 4
    return f"{year}Q{q}"

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

# =========================
# 1) LOAD
# =========================
df_cbg = pd.read_csv(PATH_CBG)
df_ap  = pd.read_csv(PATH_TAR)   
df_tc  = pd.read_csv(PATH_TC)

# Validación mínima de columnas esperadas
# Alumbrado público: energía y cargo fijo
required_ap_cols  = {"periodo", "energia_hnl_kwh", "cargo_fijo_hnl_lampara_mes"}   # <-- CAMBIO
required_cbg_cols = {"periodo", "costo_generacion_usd_mwh"}
required_tc_cols  = {"periodo", "tc_trimestre_promedio"}

missing_ap  = required_ap_cols  - set(df_ap.columns)
missing_cbg = required_cbg_cols - set(df_cbg.columns)
missing_tc  = required_tc_cols  - set(df_tc.columns)

if missing_ap:
    raise ValueError(f"Faltan columnas en alumbrado público CSV: {missing_ap}")
if missing_cbg:
    raise ValueError(f"Faltan columnas en CBG CSV: {missing_cbg}")
if missing_tc:
    raise ValueError(f"Faltan columnas en TC CSV: {missing_tc}")

# =========================
# 2) BUILD MODEL TABLE (AP)
# =========================
df = (
    df_ap[["periodo", "energia_hnl_kwh", "cargo_fijo_hnl_lampara_mes"]]   # <-- CAMBIO
    .merge(df_cbg[["periodo", "costo_generacion_usd_mwh"]], on="periodo", how="left")
    .merge(df_tc[["periodo", "tc_trimestre_promedio"]], on="periodo", how="left")
)

# Convertir CBG a HNL/kWh: (USD/MWh * HNL/USD) / 1000
df["cbg_hnl_kwh"] = (df["costo_generacion_usd_mwh"] * df["tc_trimestre_promedio"]) / 1000.0

# Ordenar por trimestre
df["period_index"] = df["periodo"].map(period_to_index)
df = df.sort_values("period_index").reset_index(drop=True)

# Lag de cargo fijo (para OLS del cargo fijo)
df["energia_hnl_kwh_lag1"] = df["energia_hnl_kwh"].shift(1)  # <-- CAMBIO

# Dataset para SARIMAX (requiere exógenas completas + y objetivo cargo fijo)
model_df = df.dropna(subset=["cbg_hnl_kwh", "tc_trimestre_promedio", "energia_hnl_kwh"]).copy()

# Dataset para OLS (requiere además lag1)
model_df_ols = model_df.dropna(subset=["energia_hnl_kwh_lag1"]).copy()

# Ajuste TEST_SIZE por si la muestra es pequeña
if len(model_df) <= TEST_SIZE + 5:
    TEST_SIZE = max(3, len(model_df)//4)

train_sar = model_df.iloc[:-TEST_SIZE].copy()
test_sar  = model_df.iloc[-TEST_SIZE:].copy()

train_ols = model_df_ols.iloc[:-TEST_SIZE].copy()
test_ols  = model_df_ols.iloc[-TEST_SIZE:].copy()

# =========================
# 3) SARIMAX (CARGO FIJO AP)  
# =========================
exog_cols = ["cbg_hnl_kwh", "tc_trimestre_promedio"]

sarimax = SARIMAX(
    train_sar["energia_hnl_kwh"],  
    exog=train_sar[exog_cols],
    order=(1, 0, 0),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

sar_pred = sarimax.get_forecast(
    steps=len(test_sar),
    exog=test_sar[exog_cols]
).predicted_mean
sar_pred = pd.Series(sar_pred.values, index=test_sar.index, name="pred_sarimax_cf")

# =========================
# 4) OLS (baseline para CARGO FIJO)  
# =========================
X_train = train_ols[["cbg_hnl_kwh", "tc_trimestre_promedio", "energia_hnl_kwh_lag1"]]
X_train = sm.add_constant(X_train)
y_train = train_ols["energia_hnl_kwh"]

ols = sm.OLS(y_train, X_train).fit()

X_test = test_ols[["cbg_hnl_kwh", "tc_trimestre_promedio", "energia_hnl_kwh_lag1"]]
X_test = sm.add_constant(X_test)
ols_pred = ols.predict(X_test)
ols_pred = pd.Series(ols_pred.values, index=test_ols.index, name="pred_ols_tap")

# =========================
# 5) METRICS (CARGO FIJO)
# =========================
sar_actual = test_sar["energia_hnl_kwh"]
ols_actual = test_ols["energia_hnl_kwh"]

metrics_df = pd.DataFrame([
    {"model": "SARIMAX(1,0,0) + exog(CBG_HNL_kWh, TC) [Tarifa AP]", **metrics(sar_actual, sar_pred), "n_test": len(sar_actual)},
    {"model": "OLS + lag1 + exog(CBG_HNL_kWh, TC) [Tarifa AP]",     **metrics(ols_actual, ols_pred), "n_test": len(ols_actual)},
])

pred_df = pd.DataFrame({
    "periodo": test_sar["periodo"].values,
    "actual_tarifa_ap_hnl/kWh": sar_actual.values,
    "pred_sarimax_energía_apa": sar_pred.values,
}).merge(
    pd.DataFrame({"periodo": test_ols["periodo"].values, "pred_ols_energia_hnl_kwh": ols_pred.values}),
    on="periodo",
    how="left"
)

metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8")
pred_df.to_csv(OUT_PREDS, index=False, encoding="utf-8")

# =========================
# 6) FORECAST HACIA ADELANTE (ESCENARIOS) + CARGO FIJO (INFLACIÓN 3%)
# =========================
last_period_index = model_df["period_index"].iloc[-1]
future_periods = [index_to_period(last_period_index + i) for i in range(1, H + 1)]

last_cbg = model_df["cbg_hnl_kwh"].iloc[-1]
last_tc  = model_df["tc_trimestre_promedio"].iloc[-1]

# ---- Inflación 3% anual aplicada a cargo fijo AP (CSV) ----
inflacion_anual = 0.030
inflacion_trimestral = (1 + inflacion_anual)**(1/4) - 1

last_cf = model_df["cargo_fijo_hnl_lampara_mes"].iloc[-1]
cf_future = [last_cf * (1 + inflacion_trimestral)**i for i in range(1, H + 1)]
cf_future = np.array(cf_future, dtype=float)

cf_infl_df = pd.DataFrame({
    "periodo": future_periods,
    "cargo_fijo_pred_hnl_lampara_mes": cf_future,
    "inflacion_anual_sup": inflacion_anual,
    "inflacion_trimestral_equiv": inflacion_trimestral,
    "metodo": "inflacion_compuesta"
})
cf_infl_df.to_csv(OUT_CF_INFL, index=False, encoding="utf-8")

# Escenarios para exógenas (los dejamos por consistencia)
scenarios = {
    "base": {
        "cbg": np.full(H, last_cbg),
        "tc":  np.full(H, last_tc),
    },
    "optimista": {
        "cbg": np.linspace(last_cbg * 0.95, last_cbg * 0.90, H),
        "tc":  np.full(H, last_tc * 0.99),
    },
    "adverso": {
        "cbg": np.linspace(last_cbg * 1.10, last_cbg * 1.20, H),
        "tc":  np.linspace(last_tc * 1.02, last_tc * 1.05, H),
    }
}

forecast_rows = []

for scen_name, sc in scenarios.items():
    exog_future = pd.DataFrame({
        "cbg_hnl_kwh": sc["cbg"],
        "tc_trimestre_promedio": sc["tc"]
    })

    fc = sarimax.get_forecast(steps=H, exog=exog_future)
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)  # IC 95%

    for i in range(H):
        forecast_rows.append({
            "escenario": scen_name,
            "periodo": future_periods[i],
            "cargo_fijo_pred_modelo": float(mean_fc.iloc[i]),
            "ic_5%": float(ci.iloc[i, 0]),
            "ic_95%": float(ci.iloc[i, 1]),
            # supuestos exógenos
            "cbg_sup": float(sc["cbg"][i]),
            "tc_sup": float(sc["tc"][i]),
            # cargo fijo por inflación (lo que quieres presentar)
            "cargo_fijo_pred_inflacion_4p5": float(cf_future[i]),
        })

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(OUT_FORECAST, index=False, encoding="utf-8")

# =========================
# PRINT RESUMEN
# =========================
print("\nOK. Exportados:")
print(" -", OUT_METRICS)
print(" -", OUT_PREDS)
print(" -", OUT_FORECAST)
print(" -", OUT_CF_INFL, "  <-- nombre solicitado")

print("\nMétricas (Cargo fijo AP):")
print(metrics_df)

print("\nPredicciones (test) - Cargo fijo AP:")
print(pred_df)

print("\nForecast escenarios (head):")
print(forecast_df.head(12))