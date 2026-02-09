import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIG 
# =========================
BASE_DIR = Path(r"C:\Users\marian.padilla\predictor_tarf")

PATH_CBG = BASE_DIR / "cree_costo_generacion_quarterly_2021_2026_from_reports.csv"
PATH_TAR = BASE_DIR / "tarifa_residencial_m_50_trimestral.csv" #cambiar según que tarifa se desea proyectar
PATH_TC  = BASE_DIR / "tc_trimestral_referencia.csv"

TEST_SIZE = 4  # últimos 4 trimestres como test
H = 4       # horizonte de forecast (trimestres) (CAMBIAR SEGÚN SE DESEA)
#CAMBIAR NOMBRES DE LOS ARCHIVOS EN LOS QUE SE GUARDAN LOS RESULTADOS SEGÚN SE NECESITE
OUT_METRICS = BASE_DIR / "model_metrics_resindencial_m_50.csv"
OUT_PREDS   = BASE_DIR / "model_predictions_resindencial_m_50.csv"
OUT_FORECAST = BASE_DIR / "forecast_tarifa_sarimax_escenarios_resindencial_m_50.csv"

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
df_tar = pd.read_csv(PATH_TAR)
df_tc  = pd.read_csv(PATH_TC)

# Validación mínima de columnas esperadas
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
df = (
    df_tar[["periodo", "energia_hnl_kwh", "cargo_fijo_hnl_mes"]]
    .merge(df_cbg[["periodo", "costo_generacion_usd_mwh"]], on="periodo", how="left")
    .merge(df_tc[["periodo", "tc_trimestre_promedio"]], on="periodo", how="left")
)

# Convertir CBG a HNL/kWh: (USD/MWh * HNL/USD) / 1000
df["cbg_hnl_kwh"] = (df["costo_generacion_usd_mwh"] * df["tc_trimestre_promedio"]) / 1000.0

# Ordenar por trimestre
df["period_index"] = df["periodo"].map(period_to_index)
df = df.sort_values("period_index").reset_index(drop=True)

# Lag tarifa (para OLS)
df["tarifa_lag1"] = df["energia_hnl_kwh"].shift(1)

# Dataset para SARIMAX (requiere exógenas completas)
model_df = df.dropna(subset=["cbg_hnl_kwh", "tc_trimestre_promedio", "energia_hnl_kwh"]).copy()

# Dataset para OLS (requiere además lag1)
model_df_ols = model_df.dropna(subset=["tarifa_lag1"]).copy()

# Ajuste TEST_SIZE por si la muestra es pequeña
if len(model_df) <= TEST_SIZE + 5:
    TEST_SIZE = max(3, len(model_df)//4)

train_sar = model_df.iloc[:-TEST_SIZE].copy()
test_sar  = model_df.iloc[-TEST_SIZE:].copy()

train_ols = model_df_ols.iloc[:-TEST_SIZE].copy()
test_ols  = model_df_ols.iloc[-TEST_SIZE:].copy()

# =========================
# 3) SARIMAX (tarifa energía)
# =========================
exog_cols = ["cbg_hnl_kwh", "tc_trimestre_promedio"]

# =========================
# 3) ANÁLISIS AIC/BIC + SELECCIÓN DE MEJOR MODELO SARIMAX
# =========================
def evaluate_sarimax_orders(train_data, exog_data, max_p=3, max_q=3, d=0):
    """
    Evalúa diferentes órdenes (p,d,q) y devuelve el mejor según AIC/BIC
    """
    results = []
    
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if p == 0 and q == 0:
                continue  # Saltar (0,0,0)
            
            try:
                model = SARIMAX(
                    train_data,
                    exog=exog_data,
                    order=(p, d, q),
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False, maxiter=200)
                
                results.append({
                    'order': f"({p},{d},{q})",
                    'AIC': model.aic,
                    'BIC': model.bic,
                    'HQIC': model.hqic,
                    'converged': model.mle_retvals['converged'],
                    'iterations': model.mle_retvals['iterations']
                })
                print(f" Order ({p},{d},{q}): AIC={model.aic:.2f}, BIC={model.bic:.2f}")
                
            except Exception as e:
                print(f" Order ({p},{d},{q}) failed: {str(e)[:50]}...")
                continue
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AIC').reset_index(drop=True)
        return results_df
    else:
        raise ValueError("No model could be fitted")
print("\n ORDENES SARIMAX CON AIC/BIC...")

# Evaluar varios órdenes
orders_df = evaluate_sarimax_orders(
    train_sar["energia_hnl_kwh"],
    train_sar[exog_cols],
    max_p=2,  # AR máximo de orden 2
    max_q=2,  # MA máximo de orden 2
    d=0       # Sin diferenciación (debido a análisis acf y adf se concluye que la serie es cuasi estacionaria)
)

print("\n RESULTADOS ORDENADOS POR AIC (menor es mejor):")
print(orders_df.to_string(index=False))

# Seleccionar el mejor modelo por AIC
best_order_str = orders_df.iloc[0]['order']
best_order = tuple(map(int, best_order_str.strip('()').split(',')))

print(f"\n MEJOR ORDEN: SARIMAX{best_order}")
print(f"   • AIC: {orders_df.iloc[0]['AIC']:.2f}")
print(f"   • BIC: {orders_df.iloc[0]['BIC']:.2f}")

# Entrenar el modelo final con el mejor orden
sarimax = SARIMAX(
    train_sar["energia_hnl_kwh"],
    exog=train_sar[exog_cols],
    order=best_order,
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False, maxiter=500)

print(f"\n Modelo final entrenado:")
print(f"   • Log-Likelihood: {sarimax.llf:.2f}")
print(f"   • AIC: {sarimax.aic:.2f}")
print(f"   • BIC: {sarimax.bic:.2f}")
print(f"   • Parámetros estimados: {len(sarimax.params)}")

#==========================

sar_pred = sarimax.get_forecast(
    steps=len(test_sar),
    exog=test_sar[exog_cols]
).predicted_mean
sar_pred = pd.Series(sar_pred.values, index=test_sar.index, name="pred_sarimax")
# =========================
# 4) OLS (baseline)
# =========================
X_train = train_ols[["cbg_hnl_kwh", "tc_trimestre_promedio", "tarifa_lag1"]]
X_train = sm.add_constant(X_train)
y_train = train_ols["energia_hnl_kwh"]

ols = sm.OLS(y_train, X_train).fit()

X_test = test_ols[["cbg_hnl_kwh", "tc_trimestre_promedio", "tarifa_lag1"]]
X_test = sm.add_constant(X_test)
ols_pred = ols.predict(X_test)
ols_pred = pd.Series(ols_pred.values, index=test_ols.index, name="pred_ols")

# =========================
# 5) METRICS
# =========================
sar_actual = test_sar["energia_hnl_kwh"]
ols_actual = test_ols["energia_hnl_kwh"]

metrics_df = pd.DataFrame([
    {"model": "SARIMAX(1,0,0) + exog(CBG_HNL_kWh, TC)", **metrics(sar_actual, sar_pred), "n_test": len(sar_actual)},
    {"model": "OLS + lag1 + exog(CBG_HNL_kWh, TC)",     **metrics(ols_actual, ols_pred), "n_test": len(ols_actual)},
])

pred_df = pd.DataFrame({
    "periodo": test_sar["periodo"].values,
    "actual_tarifa_hnl_kwh": sar_actual.values,
    "pred_sarimax_hnl_kwh": sar_pred.values,
}).merge(
    pd.DataFrame({"periodo": test_ols["periodo"].values, "pred_ols_hnl_kwh": ols_pred.values}),
    on="periodo",
    how="left"
)

metrics_df.to_csv(OUT_METRICS, index=False, encoding="utf-8")
pred_df.to_csv(OUT_PREDS, index=False, encoding="utf-8")

# =========================
# 6) FORECAST HACIA ADELANTE (ESCENARIOS) + CARGO FIJO (CONSTANTE)
# =========================
# Importante:
# - El SARIMAX necesita exógenas futuras => se hace por escenarios.
# - El cargo fijo se asume constante (último valor) por su naturaleza administrativa y se multiplica por una constante de inflación(naive ajustado)
#   para evitar tendencias espurias.
last_period_index = model_df["period_index"].iloc[-1]
future_periods = [index_to_period(last_period_index + i) for i in range(1, H + 1)]

last_cbg = model_df["cbg_hnl_kwh"].iloc[-1]
last_tc  = model_df["tc_trimestre_promedio"].iloc[-1]

# Cargo fijo con inflación (supuesto)
inflacion_anual = 0.03              # 3% anual (entre 2.5% y 3.5% basaso en observaciones de los pliegos de ajustes tarifarios de la CREE) 
inflacion_trimestral = (1 + inflacion_anual)**(1/4) - 1

last_cf = model_df["cargo_fijo_hnl_mes"].iloc[-1]
cf_future = [last_cf * (1 + inflacion_trimestral)**i for i in range(1, H + 1)]
cf_future = np.array(cf_future, dtype=float)

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
    ci = fc.conf_int(alpha=0.05)  # IC 95% (5% y 95%)

    for i in range(H):
        forecast_rows.append({
            "escenario": scen_name,
            "periodo": future_periods[i],
            "tarifa_pred_hnl_kwh": float(mean_fc.iloc[i]),
            "ic_5%": float(ci.iloc[i, 0]),
            "ic_95%": float(ci.iloc[i, 1]),
            # supuestos exógenos (transparencia)
            "cbg_sup": float(sc["cbg"][i]),
            "tc_sup": float(sc["tc"][i]),
            # cargo fijo (supuesto administrativo constante)
            "cargo_fijo_pred_hnl_mes": float(cf_future[i]),
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

print("\nMétricas:")
print(metrics_df)

print("\nPredicciones (test):")
print(pred_df)

print("\nForecast escenarios (head):")
print(forecast_df.head(12))