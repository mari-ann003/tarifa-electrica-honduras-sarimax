import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DIARIO_CSV  = BASE_DIR / "tc_diario_bch.csv"
MENSUAL_CSV = BASE_DIR / "tc_mensual_bch.csv"
OUT_QTR     = BASE_DIR / "tc_trimestral_referencia.csv"

def tc_trimestral_desde_mensual(df_m: pd.DataFrame) -> pd.DataFrame:
    df_m = df_m.copy()
    df_m["fecha"] = pd.to_datetime(df_m["fecha"], errors="coerce")
    df_m = df_m.dropna(subset=["fecha", "tc_venta"])
    df_m["periodo"] = df_m["fecha"].dt.to_period("Q")
    out = df_m.groupby("periodo")["tc_venta"].mean().reset_index()
    out["periodo"] = out["periodo"].astype(str)  # deja '2026Q1'
    return out.rename(columns={"tc_venta": "tc_trimestre_promedio"})

def tc_trimestral_desde_diario(df_d: pd.DataFrame) -> pd.DataFrame:
    df_d = df_d.copy()
    df_d["fecha"] = pd.to_datetime(df_d["fecha"], errors="coerce")
    df_d = df_d.dropna(subset=["fecha", "tc_venta"]).sort_values("fecha")
    df_d["periodo"] = df_d["fecha"].dt.to_period("Q")
    last = df_d.groupby("periodo").tail(1)[["periodo", "fecha", "tc_venta"]]
    last["periodo"] = last["periodo"].astype(str)
    return last.rename(columns={"fecha": "fecha_ref", "tc_venta": "tc_trimestre_ultimo_habil"})

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("CWD:", Path.cwd().resolve())
    print("Buscando mensual en:", MENSUAL_CSV, "->", MENSUAL_CSV.exists())
    print("Buscando diario  en:", DIARIO_CSV,  "->", DIARIO_CSV.exists())

    # --- mensual 
    if not MENSUAL_CSV.exists():
        raise FileNotFoundError(f"No encuentro {MENSUAL_CSV}. Colócalo en {BASE_DIR}")

    df_m = pd.read_csv(MENSUAL_CSV)

    # Asegura nombre de columna tc_venta
    
    if "tc_venta" not in df_m.columns:
        # intenta detectar una columna numérica típica
        cand = [c for c in df_m.columns if "venta" in c.lower()]
        if cand:
            df_m = df_m.rename(columns={cand[0]: "tc_venta"})
        else:
            raise ValueError(f"En {MENSUAL_CSV.name} no existe columna tc_venta ni una columna que contenga 'venta'.")

    q_m = tc_trimestral_desde_mensual(df_m)

    # --- diario 
    if DIARIO_CSV.exists():
        df_d = pd.read_csv(DIARIO_CSV)

        # Si no existe tc_venta, intenta renombrar
        if "tc_venta" not in df_d.columns:
            cand = [c for c in df_d.columns if "venta" in c.lower()]
            if cand:
                df_d = df_d.rename(columns={cand[0]: "tc_venta"})

        q_d = tc_trimestral_desde_diario(df_d)
        out = q_m.merge(q_d, on="periodo", how="outer").sort_values("periodo")
    else:
        out = q_m.sort_values("periodo")

    out.to_csv(OUT_QTR, index=False)
    print("OK ->", OUT_QTR.resolve())

    print(out.tail(8))
