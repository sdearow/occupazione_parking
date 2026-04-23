"""
Analisi parcheggi Roma - FCD Trips vs AC_VEI (superficie carrabile)

Steps:
  1. Carica i 4 CSV FCD e costruisce geodataframe destinazioni (EPSG:25833)
  2. Filtra destinazioni dentro il comune di Roma (spatial join con municipi)
  3. [se disponibile] Unary union di AC_VEI + buffer incrementali (0,1,2,5,10 m)
  4. [se disponibile] Classifica ogni destinazione (on-street / off-street) per buffer
  5. Analisi aggregata, oraria e per giorno della settimana
  6. [se disponibile] Stima copertura della superficie carrabile con parcheggi
  7. Salva CSV + PNG nella cartella results/

Nota: AC_VEI.shp è archiviato in Git LFS. Se il file non è disponibile localmente,
le sezioni di analisi on/off-street vengono saltate ma tutte le altre analisi vengono
eseguite ugualmente.
"""

import os
import warnings
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from shapely import wkt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET_CRS = "EPSG:25833"
BUFFER_DISTANCES = [0, 1, 2, 5, 10]
AREA_VEI_MQ = 12.5  # area occupazione stimata per veicolo (5m x 2.5m)

COLORS = {"on": "#2196F3", "off": "#F44336"}
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def is_real_shapefile(path):
    """Restituisce True se il file esiste ed è un vero shapefile (non puntatore LFS)."""
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    # I file shapefile iniziano con il magic number 0x0000270A (big-endian)
    return len(header) >= 4 and header[:4] == b'\x00\x00\x27\x0a'


# ---------------------------------------------------------------------------
# 1. Carica i 4 CSV FCD
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Caricamento CSV FCD")
print("=" * 60)
dfs = []
for i in range(1, 5):
    path = os.path.join(BASE, f"od_trips_part{i}.csv")
    dfs.append(pd.read_csv(path, low_memory=False))
trips = pd.concat(dfs, ignore_index=True)
print(f"  Totale spostamenti caricati: {len(trips):,}")

trips = trips.dropna(subset=["d", "arr_time"])
print(f"  Dopo rimozione dest. nulle:  {len(trips):,}")

trips["arr_time"] = pd.to_datetime(trips["arr_time"])
trips["hour"] = trips["arr_time"].dt.hour
trips["weekday"] = trips["arr_time"].dt.day_name()
trips["weekday_num"] = trips["arr_time"].dt.dayofweek

# ---------------------------------------------------------------------------
# 2. Costruisce GeoDataFrame destinazioni (WGS84 → EPSG:25833)
# ---------------------------------------------------------------------------
print("\nSTEP 2: Costruzione GeoDataFrame destinazioni")
trips["geom_d"] = trips["d"].apply(wkt.loads)
gdf_dest = gpd.GeoDataFrame(trips, geometry="geom_d", crs="EPSG:4326")
gdf_dest = gdf_dest.to_crs(TARGET_CRS)
print(f"  GeoDataFrame creato: {len(gdf_dest):,} punti (EPSG:25833)")

# ---------------------------------------------------------------------------
# 3. Carica municipi e filtra per comune di Roma
# ---------------------------------------------------------------------------
print("\nSTEP 3: Filtro destinazioni dentro il comune di Roma")
mun = gpd.read_file(os.path.join(BASE, "municipi_2013.shp")).to_crs(TARGET_CRS)
roma_boundary = mun.geometry.unary_union
print(f"  Municipi caricati: {len(mun)}")

mun_union = mun[["geometry"]].copy().reset_index(drop=True)
joined_roma = gpd.sjoin(gdf_dest.reset_index(), mun_union, how="inner", predicate="within")
in_roma_idx = joined_roma["index"].unique()
gdf_roma = gdf_dest.loc[in_roma_idx].copy().reset_index(drop=True)
n_total = len(gdf_dest)
n_roma = len(gdf_roma)
print(f"  Destinazioni DENTRO Roma:  {n_roma:,} / {n_total:,}  ({n_roma/n_total*100:.1f}%)")
print(f"  Destinazioni fuori Roma:   {n_total - n_roma:,} ({(n_total - n_roma)/n_total*100:.1f}%)")

filter_stats = pd.DataFrame({
    "categoria": ["Totale spostamenti FCD", "Destinazione dentro Roma", "Destinazione fuori Roma"],
    "n": [n_total, n_roma, n_total - n_roma],
    "pct": [100.0, n_roma / n_total * 100, (n_total - n_roma) / n_total * 100],
})
filter_stats.to_csv(os.path.join(RESULTS, "00_filtraggio_comune.csv"), index=False)

# ---------------------------------------------------------------------------
# 4. Analisi AC_VEI (on-street / off-street) — condizionale
# ---------------------------------------------------------------------------
AC_VEI_PATH = os.path.join(BASE, "AC_VEI.shp")
ac_vei_available = is_real_shapefile(AC_VEI_PATH)

if ac_vei_available:
    print("\nSTEP 4: Analisi on-street / off-street con AC_VEI")
    ac_vei = gpd.read_file(AC_VEI_PATH).to_crs(TARGET_CRS)
    print(f"  Poligoni AC_VEI: {len(ac_vei):,}")

    ac_vei_roma = ac_vei[ac_vei.geometry.intersects(roma_boundary)].copy()
    area_carrabile_mq = ac_vei_roma.geometry.area.sum()
    print(f"  Superficie carrabile (dentro Roma): {area_carrabile_mq / 1e6:.3f} km²")

    # Prepara GeoDataFrame punti (solo geometria + indice originale)
    pts = gdf_roma[["geometry"]].copy()
    pts.index.name = "_pt_idx"
    pts = pts.reset_index()   # colonna _pt_idx = indice originale

    print("  Classificazione on-street / off-street per ogni buffer (sjoin con R-tree) ...")
    buffer_results = []
    for buf in BUFFER_DISTANCES:
        label = f"{buf}m"
        print(f"    Buffer {label:>4} ...", end=" ", flush=True)

        # Costruisce layer AC_VEI con eventuale buffer (senza unary union)
        if buf == 0:
            ac_buf = ac_vei_roma[["geometry"]].copy().reset_index(drop=True)
        else:
            ac_buf = gpd.GeoDataFrame(
                geometry=ac_vei_roma.geometry.buffer(buf), crs=TARGET_CRS
            ).reset_index(drop=True)

        # Spatial join indicizzato (R-tree): inner → solo i punti che cadono dentro
        joined = gpd.sjoin(pts, ac_buf, how="inner", predicate="within")
        on_street_idx = joined["_pt_idx"].unique()   # indici originali on-street

        mask = pd.Series(False, index=gdf_roma.index)
        mask.loc[on_street_idx] = True

        n_on = mask.sum()
        n_off = n_roma - n_on
        pct_on = n_on / n_roma * 100
        buffer_results.append({
            "buffer_m": buf, "label": label,
            "n_on_street": int(n_on), "n_off_street": int(n_off),
            "pct_on_street": round(pct_on, 2), "pct_off_street": round(100 - pct_on, 2),
        })
        print(f"on-street {pct_on:.1f}%  |  off-street {100-pct_on:.1f}%")
        gdf_roma[f"on_street_{label}"] = mask.astype(int)

    df_buffer = pd.DataFrame(buffer_results)
    df_buffer.to_csv(os.path.join(RESULTS, "01_buffer_summary.csv"), index=False)

    # Copertura superficie carrabile
    n_on_base = int(df_buffer.loc[df_buffer.buffer_m == 0, "n_on_street"].iloc[0])
    area_occupata_mq = n_on_base * AREA_VEI_MQ
    pct_copertura = area_occupata_mq / area_carrabile_mq * 100
    coverage_stats = pd.DataFrame([{
        "superficie_carrabile_km2": round(area_carrabile_mq / 1e6, 4),
        "n_veicoli_on_street": n_on_base,
        "area_veicolo_mq_assunta": AREA_VEI_MQ,
        "area_occupata_stimata_mq": round(area_occupata_mq, 1),
        "pct_superficie_occupata": round(pct_copertura, 3),
    }])
    coverage_stats.to_csv(os.path.join(RESULTS, "04_copertura_superficie.csv"), index=False)
    print(f"  Stima area occupata: {pct_copertura:.3f}% della superficie carrabile")
else:
    print("\nSTEP 4: [SALTATO] AC_VEI.shp non disponibile come file reale (puntatore LFS).")
    print("  Esegui: git lfs pull --include='AC_VEI.shp'")
    print("  Poi rilancia lo script per ottenere l'analisi completa on/off-street.")
    df_buffer = None

# ---------------------------------------------------------------------------
# 5. Analisi temporale (usa on_street_0m se disponibile)
# ---------------------------------------------------------------------------
print("\nSTEP 5: Analisi temporale")
has_on_street = "on_street_0m" in gdf_roma.columns

day_names = {0: "Lunedì", 1: "Martedì", 2: "Mercoledì", 3: "Giovedì",
             4: "Venerdì", 5: "Sabato", 6: "Domenica"}


def temporal_stats(gdf, groupby_col, sort_col=None):
    rows = []
    for key, grp in gdf.groupby(groupby_col):
        n = len(grp)
        row = {groupby_col: key, "n_tot": n}
        if has_on_street:
            n_on = int(grp["on_street_0m"].sum())
            row.update({
                "n_on_street": n_on, "n_off_street": int(n - n_on),
                "pct_on_street": round(n_on / n * 100, 2) if n > 0 else 0,
                "pct_off_street": round((n - n_on) / n * 100, 2) if n > 0 else 0,
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    if sort_col:
        df = df.sort_values(sort_col).reset_index(drop=True)
    return df


df_hourly = temporal_stats(gdf_roma, "hour", sort_col="hour")
df_weekday = temporal_stats(gdf_roma, "weekday_num")
df_weekday["giorno"] = df_weekday["weekday_num"].map(day_names)

df_hourly.to_csv(os.path.join(RESULTS, "02_analisi_oraria.csv"), index=False)
df_weekday.to_csv(os.path.join(RESULTS, "03_analisi_giorno_settimana.csv"), index=False)

# Salva destinazioni classificate
cols_out = ["trip_id", "user_id", "dep_time", "arr_time", "hour", "weekday",
            "distance", "travel_time", "type_o", "type_d"]
if has_on_street:
    cols_out += ["on_street_0m", "on_street_1m", "on_street_2m",
                 "on_street_5m", "on_street_10m"]
cols_out = [c for c in cols_out if c in gdf_roma.columns]
gdf_roma[cols_out].to_csv(os.path.join(RESULTS, "05_destinazioni_classificate.csv"), index=False)

# ---------------------------------------------------------------------------
# 6. GRAFICI
# ---------------------------------------------------------------------------
print("\nSTEP 6: Generazione grafici")

# --- Fig 1: Distribuzione oraria arrivi ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(df_hourly["hour"], df_hourly["n_tot"], color="#90CAF9", edgecolor="white")
ax.set_xlabel("Ora del giorno (arrivo a destinazione)")
ax.set_ylabel("Numero spostamenti verso Roma")
ax.set_title("Distribuzione oraria degli arrivi nel comune di Roma\n(tutti i giorni — marzo 2023)")
ax.set_xticks(range(0, 24))
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig1_distribuzione_oraria.png"))
plt.close(fig)
print("  fig1_distribuzione_oraria.png")

# --- Fig 2: Distribuzione per giorno della settimana ---
days_sorted = df_weekday.sort_values("weekday_num")
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(days_sorted))
bars = ax.bar(x, days_sorted["n_tot"], color="#90CAF9", edgecolor="white")
for bar, n in zip(bars, days_sorted["n_tot"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{int(n):,}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(days_sorted["giorno"].tolist(), rotation=15)
ax.set_ylabel("Numero spostamenti verso Roma")
ax.set_title("Arrivi nel comune di Roma per giorno della settimana\n(marzo 2023)")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig2_distribuzione_settimanale.png"))
plt.close(fig)
print("  fig2_distribuzione_settimanale.png")

# --- Fig 3: Heatmap ora x giorno (volume) ---
pivot_n = (gdf_roma.groupby(["weekday_num", "hour"]).size()
           .unstack(level="hour").reindex(range(7)).fillna(0))
pivot_n.index = [day_names.get(i, str(i)) for i in pivot_n.index]
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(pivot_n.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(24))
ax.set_xticklabels(range(24))
ax.set_yticks(range(len(pivot_n.index)))
ax.set_yticklabels(pivot_n.index)
ax.set_xlabel("Ora del giorno")
ax.set_title("Volume di arrivi a Roma per ora e giorno della settimana")
plt.colorbar(im, ax=ax, label="n. spostamenti")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig3_heatmap_volume_ora_giorno.png"))
plt.close(fig)
print("  fig3_heatmap_volume_ora_giorno.png")

# --- Grafici AC_VEI (solo se disponibili) ---
if df_buffer is not None:
    # Fig 4: Curva cumulativa buffer
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df_buffer["buffer_m"], df_buffer["pct_on_street"],
            marker="o", linewidth=2.5, color=COLORS["on"], label="On-street")
    ax.plot(df_buffer["buffer_m"], df_buffer["pct_off_street"],
            marker="o", linewidth=2.5, color=COLORS["off"], label="Off-street")
    for _, row in df_buffer.iterrows():
        ax.annotate(f"{row.pct_on_street:.1f}%",
                    xy=(row.buffer_m, row.pct_on_street),
                    xytext=(4, 6), textcoords="offset points", fontsize=9, color=COLORS["on"])
    ax.set_xlabel("Buffer applicato alla superficie carrabile (m)")
    ax.set_ylabel("% spostamenti con destinazione in Roma")
    ax.set_title("Proporzione parcheggi on-street vs off-street\nal variare del buffer su AC_VEI")
    ax.set_xticks(BUFFER_DISTANCES)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig4_curva_buffer.png"))
    plt.close(fig)
    print("  fig4_curva_buffer.png")

    # Fig 5: Barre stacked buffer
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = df_buffer["label"].tolist()
    pct_on = df_buffer["pct_on_street"].tolist()
    pct_off = df_buffer["pct_off_street"].tolist()
    x = np.arange(len(labels))
    b1 = ax.bar(x, pct_on, color=COLORS["on"], label="On-street")
    b2 = ax.bar(x, pct_off, bottom=pct_on, color=COLORS["off"], label="Off-street")
    for bar, pct in zip(b1, pct_on):
        ax.text(bar.get_x() + bar.get_width() / 2, pct / 2,
                f"{pct:.1f}%", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    for bar, p_on, p_off in zip(b2, pct_on, pct_off):
        ax.text(bar.get_x() + bar.get_width() / 2, p_on + p_off / 2,
                f"{p_off:.1f}%", ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Buffer applicato alla superficie carrabile (m)")
    ax.set_ylabel("% spostamenti")
    ax.set_title("Distribuzione parcheggi on-street / off-street per soglia di buffer")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig5_barre_buffer.png"))
    plt.close(fig)
    print("  fig5_barre_buffer.png")

    # Fig 6: Andamento orario on/off-street
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax1, ax2 = axes
    ax1.bar(df_hourly["hour"], df_hourly["n_tot"], color="#90CAF9", edgecolor="white")
    ax1.set_ylabel("Numero spostamenti")
    ax1.set_title("Arrivi nel comune di Roma per ora del giorno")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.plot(df_hourly["hour"], df_hourly["pct_on_street"],
             marker="o", linewidth=2, color=COLORS["on"], label="% On-street")
    ax2.plot(df_hourly["hour"], df_hourly["pct_off_street"],
             marker="o", linewidth=2, color=COLORS["off"], label="% Off-street")
    ax2.set_xlabel("Ora del giorno")
    ax2.set_ylabel("%")
    ax2.set_title("% parcheggi on-street vs off-street per ora (buffer=0)")
    ax2.set_xticks(range(0, 24))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig6_andamento_orario_onoff.png"))
    plt.close(fig)
    print("  fig6_andamento_orario_onoff.png")

    # Fig 7: Giorno della settimana on/off-street
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1, ax2 = axes
    x = np.arange(len(days_sorted))
    day_labels = days_sorted["giorno"].tolist()
    ax1.bar(x, days_sorted["n_tot"], color="#90CAF9", edgecolor="white")
    ax1.set_ylabel("Numero spostamenti")
    ax1.set_title("Arrivi nel comune di Roma per giorno della settimana")
    ax1.grid(True, alpha=0.3, axis="y")
    ax2.plot(x, days_sorted["pct_on_street"], marker="o", linewidth=2,
             color=COLORS["on"], label="% On-street")
    ax2.plot(x, days_sorted["pct_off_street"], marker="o", linewidth=2,
             color=COLORS["off"], label="% Off-street")
    ax2.set_xticks(x)
    ax2.set_xticklabels(day_labels, rotation=20)
    ax2.set_ylabel("%")
    ax2.set_title("% parcheggi on-street vs off-street per giorno (buffer=0)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig7_andamento_settimanale_onoff.png"))
    plt.close(fig)
    print("  fig7_andamento_settimanale_onoff.png")

    # Fig 8: Heatmap on-street % ora x giorno
    pivot_on = (
        gdf_roma.groupby(["weekday_num", "hour"])["on_street_0m"]
        .apply(lambda x: x.sum() / len(x) * 100 if len(x) > 0 else 0)
        .unstack(level="hour").reindex(range(7)).fillna(0)
    )
    pivot_on.index = [day_names.get(i, str(i)) for i in pivot_on.index]
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot_on.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=100)
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(len(pivot_on.index)))
    ax.set_yticklabels(pivot_on.index)
    ax.set_xlabel("Ora del giorno")
    ax.set_title("% parcheggi on-street (buffer=0) per ora e giorno della settimana")
    plt.colorbar(im, ax=ax, label="% on-street")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig8_heatmap_onstreet_ora_giorno.png"))
    plt.close(fig)
    print("  fig8_heatmap_onstreet_ora_giorno.png")

# ---------------------------------------------------------------------------
# 7. Riepilogo finale
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RIEPILOGO ANALISI PARCHEGGI ROMA")
print("=" * 60)
print(f"Spostamenti FCD totali:          {n_total:>10,}")
print(f"Destinazioni dentro Roma:        {n_roma:>10,}  ({n_roma/n_total*100:.1f}%)")
print(f"Destinazioni fuori Roma:         {n_total-n_roma:>10,}  ({(n_total-n_roma)/n_total*100:.1f}%)")

if df_buffer is not None:
    print(f"\n{'Buffer':>8} | {'On-street':>12} | {'Off-street':>12} | {'% On':>7} | {'% Off':>7}")
    print("-" * 56)
    for _, r in df_buffer.iterrows():
        print(f"{r.label:>8} | {r.n_on_street:>12,} | {r.n_off_street:>12,} | "
              f"{r.pct_on_street:>6.1f}% | {r.pct_off_street:>6.1f}%")
else:
    print("\n[Analisi on/off-street non disponibile: AC_VEI.shp mancante]")
    print("Per abilitarla: git lfs pull --include='AC_VEI.shp' && python3 analyze_parking.py")

print("\n" + "=" * 60)
print(f"Output salvati in: {RESULTS}/")
