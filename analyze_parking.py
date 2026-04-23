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

trips["dep_time"] = pd.to_datetime(trips["dep_time"])
trips["arr_time"] = pd.to_datetime(trips["arr_time"])
trips["hour"] = trips["arr_time"].dt.hour
trips["weekday"] = trips["arr_time"].dt.day_name()
trips["weekday_num"] = trips["arr_time"].dt.dayofweek

# ---------------------------------------------------------------------------
# 2. Costruisce GeoDataFrame destinazioni (WGS84 → EPSG:25833)
# ---------------------------------------------------------------------------
print("\nSTEP 2: Costruzione GeoDataFrame destinazioni")
trips["geometry"] = trips["d"].apply(wkt.loads)
gdf_dest = gpd.GeoDataFrame(trips, geometry="geometry", crs="EPSG:4326")
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
# 7. Stima veicoli parcheggiati vs in uso (snapshot orari)
# ---------------------------------------------------------------------------
print("\nSTEP 7: Stima veicoli parcheggiati vs in uso")

# Usa tutti gli spostamenti (non solo dest. in Roma):
# vogliamo capire quanta parte della flotta osservata è ferma in ogni ora.
t_all = trips.copy()
t_all["date"] = t_all["dep_time"].dt.date

# Teniamo solo viaggi che iniziano e finiscono nello stesso giorno
# (i cross-mezzanotte sono meno dell'1% e complicano il calcolo)
same_day = t_all["dep_time"].dt.date == t_all["arr_time"].dt.date
t_all = t_all[same_day].copy()

t_all["dep_min"] = t_all["dep_time"].dt.hour * 60 + t_all["dep_time"].dt.minute
t_all["arr_min"] = t_all["arr_time"].dt.hour * 60 + t_all["arr_time"].dt.minute

# Numero di veicoli attivi per giorno (almeno uno spostamento)
daily_active = t_all.groupby("date")["user_id"].nunique()

# Mappa data → giorno settimana
date_weekday = {d: pd.Timestamp(d).dayofweek for d in daily_active.index}

print(f"  Giorni coperti: {len(daily_active)}  |  "
      f"Veicoli unici: {t_all['user_id'].nunique():,}")

# Snapshot orario: per ogni ora h contiamo i veicoli in movimento alle h:30
print("  Calcolo snapshot orari ...", end=" ", flush=True)
snap_records = []
for hour in range(24):
    snap_min = hour * 60 + 30
    active_now = t_all[(t_all["dep_min"] <= snap_min) & (t_all["arr_min"] >= snap_min)]
    moving_by_date = active_now.groupby("date")["user_id"].nunique()

    for date, n_active_day in daily_active.items():
        n_moving = int(moving_by_date.get(date, 0))
        n_parked = n_active_day - n_moving
        snap_records.append({
            "date": date,
            "weekday_num": date_weekday[date],
            "hour": hour,
            "n_active": int(n_active_day),
            "n_moving": n_moving,
            "n_parked": n_parked,
            "pct_parked": round(n_parked / n_active_day * 100, 2),
            "pct_moving": round(n_moving / n_active_day * 100, 2),
        })
print("fatto.")

df_snap = pd.DataFrame(snap_records)

# Media per ora del giorno (su tutti i giorni)
df_hourly_usage = (
    df_snap.groupby("hour")
    .agg(
        avg_n_active=("n_active", "mean"),
        avg_n_moving=("n_moving", "mean"),
        avg_n_parked=("n_parked", "mean"),
        avg_pct_moving=("pct_moving", "mean"),
        avg_pct_parked=("pct_parked", "mean"),
    )
    .round(2)
    .reset_index()
)

# Media per giorno settimana × ora (per heatmap)
df_heatmap_usage = (
    df_snap.groupby(["weekday_num", "hour"])["pct_parked"]
    .mean().round(2)
    .unstack("hour")
    .reindex(range(7))
)
df_heatmap_usage.index = [day_names.get(i, str(i)) for i in df_heatmap_usage.index]

# Stima notturna (ore 23-05)
night_hours = [23, 0, 1, 2, 3, 4, 5]
night_df = df_snap[df_snap["hour"].isin(night_hours)]
avg_pct_parked_night = night_df["pct_parked"].mean()
avg_pct_parked_day = df_snap[~df_snap["hour"].isin(night_hours)]["pct_parked"].mean()
avg_pct_parked_all = df_snap["pct_parked"].mean()

peak_moving_hour = int(df_hourly_usage.loc[df_hourly_usage["avg_pct_moving"].idxmax(), "hour"])
peak_moving_pct = df_hourly_usage["avg_pct_moving"].max()

print(f"  Media giornaliera % veicoli parcheggiati: {avg_pct_parked_all:.1f}%")
print(f"  Media diurna (06-22)  % veicoli parcheggiati: {avg_pct_parked_day:.1f}%")
print(f"  Media notturna (23-05) % veicoli parcheggiati: {avg_pct_parked_night:.1f}%")
print(f"  Ora con più veicoli in movimento: {peak_moving_hour}:30  ({peak_moving_pct:.1f}% in uso)")

# Salva CSV
df_hourly_usage.to_csv(os.path.join(RESULTS, "06_uso_veicoli_orario.csv"), index=False)
df_snap.groupby("weekday_num").agg(
    avg_pct_parked=("pct_parked", "mean"),
    avg_pct_moving=("pct_moving", "mean"),
).round(2).reset_index().assign(
    giorno=lambda d: d["weekday_num"].map(day_names)
).to_csv(os.path.join(RESULTS, "07_uso_veicoli_per_giorno.csv"), index=False)

summary_uso = pd.DataFrame([{
    "media_pct_parcheggiati_giornaliero": round(avg_pct_parked_all, 2),
    "media_pct_parcheggiati_diurno_06_22": round(avg_pct_parked_day, 2),
    "media_pct_parcheggiati_notturno_23_05": round(avg_pct_parked_night, 2),
    "ora_picco_uso": peak_moving_hour,
    "pct_in_uso_al_picco": round(float(peak_moving_pct), 2),
}])
summary_uso.to_csv(os.path.join(RESULTS, "08_sintesi_uso_flotta.csv"), index=False)

# --- Fig 9: % parcheggiati vs in uso per ora ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(df_hourly_usage["hour"], df_hourly_usage["avg_pct_parked"],
                alpha=0.35, color="#F44336", label="_nolegend_")
ax.fill_between(df_hourly_usage["hour"], df_hourly_usage["avg_pct_moving"],
                alpha=0.35, color="#2196F3", label="_nolegend_")
ax.plot(df_hourly_usage["hour"], df_hourly_usage["avg_pct_parked"],
        marker="o", linewidth=2, color="#F44336", label="% Parcheggiati")
ax.plot(df_hourly_usage["hour"], df_hourly_usage["avg_pct_moving"],
        marker="o", linewidth=2, color="#2196F3", label="% In uso (in viaggio)")
# Annotazione picco
ax.annotate(f"Picco\n{peak_moving_pct:.1f}%",
            xy=(peak_moving_hour, float(peak_moving_pct)),
            xytext=(peak_moving_hour + 1.2, float(peak_moving_pct) - 4),
            arrowprops=dict(arrowstyle="->", color="#2196F3"), color="#2196F3", fontsize=9)
ax.axhspan(0, 100, xmin=23/24, alpha=0.08, color="navy", label="Fascia notturna")
ax.axhspan(0, 100, xmin=0, xmax=5/24, alpha=0.08, color="navy")
ax.set_xlabel("Ora del giorno")
ax.set_ylabel("%")
ax.set_title("% veicoli parcheggiati vs in uso per ora del giorno\n"
             "(flotta osservata — media marzo 2023, snapshot alle :30)")
ax.set_xticks(range(0, 24))
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_ylim(0, 105)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig9_pct_parcheggiati_vs_inuso_orario.png"))
plt.close(fig)
print("  fig9_pct_parcheggiati_vs_inuso_orario.png")

# --- Fig 10: Numero assoluto medio veicoli parcheggiati vs in uso ---
fig, ax = plt.subplots(figsize=(12, 5))
ax.stackplot(df_hourly_usage["hour"],
             df_hourly_usage["avg_n_moving"],
             df_hourly_usage["avg_n_parked"],
             labels=["In uso (in viaggio)", "Parcheggiati"],
             colors=["#2196F3", "#F44336"], alpha=0.8)
ax.set_xlabel("Ora del giorno")
ax.set_ylabel("N. veicoli (media giornaliera)")
ax.set_title("Numero medio di veicoli parcheggiati vs in uso per ora\n"
             "(flotta osservata — marzo 2023)")
ax.set_xticks(range(0, 24))
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig10_n_veicoli_parcheggiati_vs_inuso.png"))
plt.close(fig)
print("  fig10_n_veicoli_parcheggiati_vs_inuso.png")

# --- Fig 11: Heatmap % parcheggiati ora x giorno settimana ---
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(df_heatmap_usage.values, aspect="auto", cmap="RdYlGn_r",
               vmin=50, vmax=100)
ax.set_xticks(range(24))
ax.set_xticklabels(range(24))
ax.set_yticks(range(len(df_heatmap_usage.index)))
ax.set_yticklabels(df_heatmap_usage.index)
ax.set_xlabel("Ora del giorno")
ax.set_title("% veicoli parcheggiati per ora e giorno della settimana\n"
             "(verde = più in uso, rosso = più parcheggiati)")
plt.colorbar(im, ax=ax, label="% parcheggiati")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig11_heatmap_pct_parcheggiati.png"))
plt.close(fig)
print("  fig11_heatmap_pct_parcheggiati.png")

# --- Fig 12: Confronto giorno / notte / media ---
labels_comp = ["Media\ngiornaliera", "Diurno\n(06–22)", "Notturno\n(23–05)"]
values_comp = [avg_pct_parked_all, avg_pct_parked_day, avg_pct_parked_night]
colors_comp = ["#9C27B0", "#FF9800", "#3F51B5"]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels_comp, values_comp, color=colors_comp, width=0.5)
for bar, val in zip(bars, values_comp):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylabel("% veicoli parcheggiati")
ax.set_title("% media veicoli parcheggiati:\nconfronto giornaliero vs fasce orarie")
ax.set_ylim(0, 105)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig12_confronto_giorno_notte.png"))
plt.close(fig)
print("  fig12_confronto_giorno_notte.png")

# ---------------------------------------------------------------------------
# 8. Riepilogo finale
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

print(f"\n  % veicoli parcheggiati — media: {avg_pct_parked_all:.1f}%  "
      f"| diurno: {avg_pct_parked_day:.1f}%  | notturno: {avg_pct_parked_night:.1f}%")

# ---------------------------------------------------------------------------
# 9. Stime a scala di città (calibrazione su traffico reale di picco)
# ---------------------------------------------------------------------------
print("\nSTEP 9: Stime a scala di città")

# --- Parametro di calibrazione ---
# Veicoli unici che effettuano almeno uno spostamento nella fascia di punta mattutina.
# Modifica questi valori se disponi di un riferimento aggiornato.
REF_PEAK_VEHICLES  = 300_000   # veicoli unici con almeno un viaggio 07:00–09:00
PEAK_WINDOW_START  = 7         # ora inizio fascia (inclusa)
PEAK_WINDOW_END    = 9         # ora fine fascia (esclusa)

# Stessa metrica nel campione FCD: media giornaliera di veicoli unici
# con almeno una partenza nella finestra 07:00–09:00
peak_trips_sample = t_all[
    (t_all["dep_time"].dt.hour >= PEAK_WINDOW_START) &
    (t_all["dep_time"].dt.hour <  PEAK_WINDOW_END)
]
avg_unique_peak_sample = peak_trips_sample.groupby("date")["user_id"].nunique().mean()

SCALE = REF_PEAK_VEHICLES / avg_unique_peak_sample
implied_daily_fleet = df_hourly_usage["avg_n_active"].iloc[0] * SCALE

peak_hour = int(df_hourly_usage.loc[df_hourly_usage["avg_pct_moving"].idxmax(), "hour"])

print(f"  Riferimento: veicoli unici con partenza {PEAK_WINDOW_START}:00–{PEAK_WINDOW_END}:00: "
      f"{REF_PEAK_VEHICLES:,}")
print(f"  Campione FCD (media/giorno stessa finestra):  {avg_unique_peak_sample:,.0f} veicoli")
print(f"  Fattore di espansione:                        {SCALE:.1f}×")
print(f"  Flotta giornaliera attiva stimata:            {implied_daily_fleet:,.0f} veicoli")

# Parco veicolare totale registrato nel Comune di Roma (fonte: ACI).
# Include tutti i veicoli — anche quelli che non si spostano quotidianamente.
# Modifica questo valore se disponi di un dato aggiornato.
FLEET_TOTAL_ROME = 1_600_000

print(f"  Parco veicolare totale (ACI, modif.):         {FLEET_TOTAL_ROME:,} veicoli")

# --- Espansione del profilo orario ---
# city_n_moving: veicoli in moto per ogni ora (scala campione → città)
# city_n_parked_active: veicoli della flotta attiva che sono fermi
# city_n_parked_total: TUTTI i veicoli non in moto (flotta totale - in moto)
#   → questo è il denominatore corretto per la superficie occupata,
#     perché include anche i veicoli che non si muovono mai quel giorno.
df_city = df_hourly_usage.copy()
df_city["city_n_moving"]        = (df_city["avg_n_moving"] * SCALE).round(0).astype(int)
df_city["city_n_parked_active"] = (df_city["avg_n_parked"] * SCALE).round(0).astype(int)
df_city["city_n_parked_total"]  = FLEET_TOTAL_ROME - df_city["city_n_moving"]
df_city["pct_moving_total"]     = (df_city["city_n_moving"] / FLEET_TOTAL_ROME * 100).round(2)
df_city["pct_parked_total"]     = (df_city["city_n_parked_total"] / FLEET_TOTAL_ROME * 100).round(2)

# --- On/off-street ---
# La % on/off-street viene dall'analisi AC_VEI sui trip FCD (veicoli che hanno fatto
# almeno un viaggio). Applicata alla flotta totale parcheggiata è un'approssimazione:
# i veicoli che non si spostano mai tendono ad essere più spesso in garage (off-street),
# quindi le stime on-street sono da considerarsi un limite superiore.
if df_buffer is not None:
    pct_on_0m  = float(df_buffer.loc[df_buffer.buffer_m == 0,  "pct_on_street"].iloc[0]) / 100
    pct_on_2m  = float(df_buffer.loc[df_buffer.buffer_m == 2,  "pct_on_street"].iloc[0]) / 100
    pct_on_5m  = float(df_buffer.loc[df_buffer.buffer_m == 5,  "pct_on_street"].iloc[0]) / 100
    pct_on_10m = float(df_buffer.loc[df_buffer.buffer_m == 10, "pct_on_street"].iloc[0]) / 100
else:
    pct_on_0m, pct_on_2m, pct_on_5m, pct_on_10m = 0.650, 0.764, 0.843, 0.901
    print("  [% on/off-street da analisi precedente: 0m=65%, 2m=76.4%, 5m=84.3%, 10m=90.1%]")

df_city["city_n_onstreet_0m"]  = (df_city["city_n_parked_total"] * pct_on_0m).round(0).astype(int)
df_city["city_n_offstreet_0m"] = df_city["city_n_parked_total"] - df_city["city_n_onstreet_0m"]
df_city["city_n_onstreet_2m"]  = (df_city["city_n_parked_total"] * pct_on_2m).round(0).astype(int)
df_city["city_n_offstreet_2m"] = df_city["city_n_parked_total"] - df_city["city_n_onstreet_2m"]

df_city.to_csv(os.path.join(RESULTS, "09_stima_citta_oraria.csv"), index=False)

# --- Sintesi per fascia ---
SURF_KM2 = 135.142
night_city = df_city[df_city["hour"].isin(night_hours)]
day_city   = df_city[~df_city["hour"].isin(night_hours)]

def city_summary(df_sub, label):
    n_m    = df_sub["city_n_moving"].mean()
    n_pt   = df_sub["city_n_parked_total"].mean()
    n_on0  = df_sub["city_n_onstreet_0m"].mean()
    n_on2  = df_sub["city_n_onstreet_2m"].mean()
    area0  = n_on0 * AREA_VEI_MQ / 1e6
    area2  = n_on2 * AREA_VEI_MQ / 1e6
    return {
        "fascia": label,
        "n_in_movimento": int(round(n_m)),
        "pct_in_movimento": round(n_m / FLEET_TOTAL_ROME * 100, 1),
        "n_parcheggiati_totale": int(round(n_pt)),
        "pct_parcheggiati": round(n_pt / FLEET_TOTAL_ROME * 100, 1),
        "n_on_street_0m": int(round(n_on0)),
        "n_off_street_0m": int(round(n_pt - n_on0)),
        "n_on_street_2m": int(round(n_on2)),
        "n_off_street_2m": int(round(n_pt - n_on2)),
        "superficie_on_street_km2_0m": round(area0, 2),
        "superficie_on_street_km2_2m": round(area2, 2),
        "pct_superficie_carrabile_0m": round(area0 / SURF_KM2 * 100, 1),
        "pct_superficie_carrabile_2m": round(area2 / SURF_KM2 * 100, 1),
    }

rows_summary = [
    city_summary(df_city,     "24h (media giornaliera)"),
    city_summary(day_city,    "diurno 06-22"),
    city_summary(night_city,  "notturno 23-05"),
    city_summary(df_city[df_city["hour"] == peak_hour], f"picco ({peak_hour}:30)"),
]
df_city_summary = pd.DataFrame(rows_summary)
df_city_summary.to_csv(os.path.join(RESULTS, "10_stima_citta_sintesi.csv"), index=False)

# Stampa riepilogo
print(f"\n  {'Fascia':<26} {'In moto':>9} {'% moto':>7} "
      f"{'Parcheg.':>10} {'% parch.':>9} {'On-str 2m':>11} {'Superf. 2m':>11}")
print("  " + "-" * 90)
for r in rows_summary:
    print(f"  {r['fascia']:<26} {r['n_in_movimento']:>9,} {r['pct_in_movimento']:>6.1f}% "
          f"{r['n_parcheggiati_totale']:>10,} {r['pct_parcheggiati']:>8.1f}% "
          f"{r['n_on_street_2m']:>11,} {r['pct_superficie_carrabile_2m']:>9.1f}%")

# --- Fig 13: Profilo orario a scala di città ---
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
ax1, ax2 = axes

ax1.stackplot(df_city["hour"],
              df_city["city_n_moving"],
              df_city["city_n_onstreet_2m"],
              df_city["city_n_offstreet_2m"],
              labels=["In movimento", "Parcheggiati on-street (≤2m)", "Parcheggiati off-street"],
              colors=["#2196F3", "#FF9800", "#F44336"], alpha=0.85)
ax1.set_ylabel(f"N. veicoli (su {FLEET_TOTAL_ROME:,} totali)")
ax1.set_title(f"Stima veicoli in movimento e in sosta a Roma per ora del giorno\n"
              f"(parco totale {FLEET_TOTAL_ROME:,} — buffer 2m — calibrato su {REF_PEAK_VEHICLES:,} veh. in punta 07–09)")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(True, alpha=0.25, axis="y")

ax2.plot(df_city["hour"], df_city["pct_parked_total"],
         marker="o", linewidth=2, color="#F44336", label="% Parcheggiati (su parco totale)")
ax2.plot(df_city["hour"], df_city["pct_moving_total"],
         marker="o", linewidth=2, color="#2196F3", label="% In movimento (su parco totale)")
ax2.set_xlabel("Ora del giorno")
ax2.set_ylabel("%")
ax2.set_title("% veicoli parcheggiati vs in movimento (su parco veicolare totale)")
ax2.set_xticks(range(0, 24))
ax2.yaxis.set_major_formatter(mticker.PercentFormatter())
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig13_stima_citta_profilo_orario.png"))
plt.close(fig)
print("  fig13_stima_citta_profilo_orario.png")

# --- Fig 14: Barre confronto fasce orarie (stima città) ---
fasce   = ["24h\nmedia", f"Diurno\n06–22", "Notturno\n23–05", f"Picco\n{peak_hour}:30"]
n_on2   = [r["n_on_street_2m"]  for r in rows_summary]
n_off2  = [r["n_off_street_2m"] for r in rows_summary]
n_mov   = [r["n_in_movimento"]  for r in rows_summary]
x       = np.arange(len(fasce))

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x, n_on2,  color="#FF9800", label="Parcheggiati on-street (≤2m)")
b2 = ax.bar(x, n_off2, bottom=n_on2,  color="#F44336", label="Parcheggiati off-street")
b3 = ax.bar(x, n_mov,  bottom=[a+b for a,b in zip(n_on2, n_off2)],
            color="#2196F3", label="In movimento")
for bar, val in zip(b1, n_on2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
            f"{val:,.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
for bar, base, val in zip(b2, n_on2, n_off2):
    ax.text(bar.get_x()+bar.get_width()/2, base + val/2,
            f"{val:,.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
for bar, base_on, base_off, val in zip(b3, n_on2, n_off2, n_mov):
    ax.text(bar.get_x()+bar.get_width()/2, base_on+base_off + val/2,
            f"{val:,.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(fasce)
ax.set_ylabel("N. veicoli stimati")
ax.set_title(f"Distribuzione veicoli a Roma per fascia oraria\n"
             f"(calibrato su {REF_PEAK_VEHICLES:,} veicoli unici in punta 07–09, buffer 2m)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig14_stima_citta_fasce_orarie.png"))
plt.close(fig)
print("  fig14_stima_citta_fasce_orarie.png")

print("\n" + "=" * 60)
print(f"Output salvati in: {RESULTS}/")
