"""
Mappe di densità e pressione di parcheggio — Comune di Roma
============================================================
Produce:
  Mappe città-wide (griglia esagonale 250 m):
    map01  Densità sosta totale (soste/km²)
    map02  Densità sosta diurna  07–20
    map03  Densità sosta notturna 20–07
    map04  Indice residenziale (% soste notturne)

  Mappe per municipio (choropleth):
    map05  Densità sosta totale
    map06  Densità sosta diurna
    map07  Densità sosta notturna
    map08  Indice residenziale
    map09  Bar chart indicatori per municipio

  Se AC_VEI disponibile (on_street_2m nella col. 05_destinazioni):
    map10  % soste on-street per municipio
    map11  Saturazione superficie carrabile per municipio
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from shapely import wkt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
MAPS    = os.path.join(RESULTS, "maps")
os.makedirs(MAPS, exist_ok=True)

TARGET_CRS   = "EPSG:25833"
HEX_SIZE_M   = 250        # circumradius esagono in metri
DAY_HOURS    = range(7, 20)
AREA_VEI_MQ  = 12.5

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# ── Helpers ───────────────────────────────────────────────────────────────────
def is_real_shapefile(path):
    if not os.path.exists(path): return False
    with open(path, "rb") as f: h = f.read(4)
    return h == b'\x00\x00\x27\x0a'


def make_hex_grid(bounds, size, crs):
    """Griglia esagonale pointy-top con circumradius=size (metri)."""
    xmin, ymin, xmax, ymax = bounds
    w = size * np.sqrt(3)
    row_step = size * 1.5
    hexes = []
    row = 0
    y = ymin - size
    while y < ymax + size:
        x0 = xmin - w + (w / 2 if row % 2 else 0)
        x = x0
        while x < xmax + w:
            angles = np.radians(np.arange(30, 391, 60))
            verts = [(x + size * np.cos(a), y + size * np.sin(a)) for a in angles]
            hexes.append(Polygon(verts))
            x += w
        y += row_step
        row += 1
    grid = gpd.GeoDataFrame({"hex_id": range(len(hexes))},
                             geometry=hexes, crs=crs)
    return grid


def save_map(fig, name):
    path = os.path.join(MAPS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def choropleth(gdf, col, title, cmap, fname,
               mun_bounds=None, vmin=None, vmax=None,
               label="", fmt="{:.0f}"):
    """Genera un choropleth per municipio."""
    vmin = vmin if vmin is not None else gdf[col].quantile(0.05)
    vmax = vmax if vmax is not None else gdf[col].quantile(0.95)
    fig, ax = plt.subplots(figsize=(12, 11))
    ax.set_facecolor("#e8e8e8")
    gdf.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax,
             edgecolor="#555", linewidth=1.2, ax=ax, legend=False)
    if mun_bounds is not None:
        mun_bounds.boundary.plot(ax=ax, color="#222", linewidth=0.6)
    # Label municipio
    for _, row in gdf.iterrows():
        centroid = row.geometry.centroid
        val = row[col]
        ax.annotate(f"M{int(row['Numero'])}\n{fmt.format(val)}",
                    xy=(centroid.x, centroid.y),
                    ha="center", va="center", fontsize=7.5,
                    color="white",
                    fontweight="bold",
                    path_effects=[
                        matplotlib.patheffects.withStroke(
                            linewidth=2, foreground="#333")
                    ])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_axis_off()
    save_map(fig, fname)


# ── 1. Carica FCD + filtro Roma ───────────────────────────────────────────────
print("Caricamento dati FCD ...")
dfs = []
for i in range(1, 5):
    dfs.append(pd.read_csv(os.path.join(BASE, f"od_trips_part{i}.csv"),
                           low_memory=False))
trips = pd.concat(dfs, ignore_index=True)
trips = trips.dropna(subset=["d", "arr_time"])
trips["arr_time"]   = pd.to_datetime(trips["arr_time"])
trips["dep_time"]   = pd.to_datetime(trips["dep_time"])
trips["hour"]       = trips["arr_time"].dt.hour
trips["weekday_num"]= trips["arr_time"].dt.dayofweek
trips["geometry"]   = trips["d"].apply(wkt.loads)

gdf_dest = gpd.GeoDataFrame(trips, geometry="geometry", crs="EPSG:4326"
                             ).to_crs(TARGET_CRS)

mun = gpd.read_file(os.path.join(BASE, "municipi_2013.shp")).to_crs(TARGET_CRS)
mun_union_gdf = mun[["geometry"]].copy().reset_index(drop=True)

print("  Filtro Roma ...")
joined_roma = gpd.sjoin(gdf_dest.reset_index(), mun_union_gdf,
                        how="inner", predicate="within")
gdf_roma = gdf_dest.loc[joined_roma["index"].unique()].copy().reset_index(drop=True)
n_roma = len(gdf_roma)
print(f"  Destinazioni dentro Roma: {n_roma:,}")

# ── 2. Merge colonne on/off-street (se già calcolate) ─────────────────────────
class_csv = os.path.join(RESULTS, "05_destinazioni_classificate.csv")
has_onstreet = False
if os.path.exists(class_csv):
    cls = pd.read_csv(class_csv, usecols=lambda c: c in
                      ["trip_id", "on_street_0m", "on_street_2m"])
    if "on_street_2m" in cls.columns and "trip_id" in cls.columns:
        gdf_roma = gdf_roma.merge(cls, on="trip_id", how="left")
        has_onstreet = "on_street_2m" in gdf_roma.columns
print(f"  Dati on/off-street disponibili: {has_onstreet}")

# ── 3. Split giorno / notte ───────────────────────────────────────────────────
gdf_day   = gdf_roma[gdf_roma["hour"].isin(DAY_HOURS)].copy()
gdf_night = gdf_roma[~gdf_roma["hour"].isin(DAY_HOURS)].copy()
print(f"  Sosta diurna:   {len(gdf_day):,}  |  notturna: {len(gdf_night):,}")

# ── 4. Tag municipio su ogni destinazione ─────────────────────────────────────
print("  Tagging municipio per destinazione ...")
mun_tag = mun[["Numero", "Name", "geometry"]].copy().reset_index(drop=True)
gdf_tagged = gpd.sjoin(gdf_roma[["geometry", "trip_id", "hour",
                                  "weekday_num"] +
                                 (["on_street_2m"] if has_onstreet else [])],
                        mun_tag, how="left", predicate="within")

# ── 5. Statistiche per municipio ──────────────────────────────────────────────
print("  Calcolo statistiche per municipio ...")
mun_stats = mun_tag.copy()

def count_stops(gdf, mun_col="Numero"):
    return gdf.groupby(mun_col).size().rename("n")

tot_by_mun   = count_stops(gdf_tagged)
day_by_mun   = count_stops(gdf_tagged[gdf_tagged["hour"].isin(DAY_HOURS)])
night_by_mun = count_stops(gdf_tagged[~gdf_tagged["hour"].isin(DAY_HOURS)])

mun_stats = mun_stats.join(
    pd.DataFrame({"n_tot": tot_by_mun,
                  "n_day": day_by_mun,
                  "n_night": night_by_mun}),
    on="Numero"
).fillna(0)

mun_stats["area_km2"]       = mun_stats.geometry.area / 1e6
mun_stats["dens_tot"]       = mun_stats["n_tot"]   / mun_stats["area_km2"]
mun_stats["dens_day"]       = mun_stats["n_day"]   / mun_stats["area_km2"]
mun_stats["dens_night"]     = mun_stats["n_night"] / mun_stats["area_km2"]
mun_stats["pct_night"]      = (mun_stats["n_night"] /
                                mun_stats["n_tot"].replace(0, np.nan) * 100)

if has_onstreet:
    on2_by_mun = (gdf_tagged[gdf_tagged["on_street_2m"] == 1]
                  .groupby("Numero").size().rename("n_onstreet_2m"))
    mun_stats = mun_stats.join(on2_by_mun, on="Numero").fillna(0)
    mun_stats["pct_onstreet_2m"] = (mun_stats["n_onstreet_2m"] /
                                     mun_stats["n_tot"].replace(0, np.nan) * 100)

mun_stats.drop(columns="geometry").to_csv(
    os.path.join(RESULTS, "11_statistiche_municipio.csv"), index=False)

# ── 6. Griglia esagonale ──────────────────────────────────────────────────────
print("Generazione griglia esagonale ...")
bounds = mun.total_bounds  # [xmin, ymin, xmax, ymax]
hexgrid = make_hex_grid(bounds, HEX_SIZE_M, TARGET_CRS)
roma_poly = mun.geometry.unary_union
hexgrid = hexgrid[hexgrid.geometry.intersects(roma_poly)].copy()
hexgrid["area_km2"] = hexgrid.geometry.area / 1e6

def join_counts(pts, hexgrid, label):
    j = gpd.sjoin(pts[["geometry"]], hexgrid[["hex_id", "geometry"]],
                  how="inner", predicate="within")
    c = j.groupby("hex_id").size().rename(label)
    return hexgrid.join(c, on="hex_id")

print("  Spatial join punti → celle ...")
hex_tot   = join_counts(gdf_roma,  hexgrid, "n_tot")
hex_day   = join_counts(gdf_day,   hexgrid, "n_day")
hex_night = join_counts(gdf_night, hexgrid, "n_night")

hex_map = hex_tot[["hex_id", "area_km2", "geometry", "n_tot"]].copy()
hex_map = hex_map.join(hex_day  [["hex_id", "n_day"  ]].set_index("hex_id"), on="hex_id")
hex_map = hex_map.join(hex_night[["hex_id", "n_night"]].set_index("hex_id"), on="hex_id")
hex_map = hex_map.fillna(0)

hex_map["dens_tot"]   = hex_map["n_tot"]   / hex_map["area_km2"]
hex_map["dens_day"]   = hex_map["n_day"]   / hex_map["area_km2"]
hex_map["dens_night"] = hex_map["n_night"] / hex_map["area_km2"]
hex_map["pct_night"]  = np.where(
    hex_map["n_tot"] > 0,
    hex_map["n_night"] / hex_map["n_tot"] * 100, np.nan)

# ── 7. Funzione mappa esagonale ───────────────────────────────────────────────
import matplotlib.patheffects as pe

def hex_map_plot(hexdf, col, title, cmap, fname,
                 mun_gdf=None, label="", pct_clip=97, min_n=3):
    data = hexdf[hexdf["n_tot"] >= min_n].copy()
    vmax = np.nanpercentile(data[col].values, pct_clip)
    vmin = 0
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_facecolor("#d0d0d0")
    # Roma fill
    mun.plot(ax=ax, color="#f0f0f0", edgecolor="none")
    # Hex
    data.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax,
              edgecolor="none", linewidth=0, ax=ax, alpha=0.92, legend=False)
    # Confini municipio
    if mun_gdf is not None:
        mun_gdf.boundary.plot(ax=ax, color="#444", linewidth=1.2)
        for _, r in mun_gdf.iterrows():
            ax.annotate(f"M{int(r['Numero'])}",
                        xy=(r.geometry.centroid.x, r.geometry.centroid.y),
                        ha="center", va="center", fontsize=7, color="#222",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cbar.set_label(label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_axis_off()
    save_map(fig, fname)


# ── 8. Mappe città-wide ───────────────────────────────────────────────────────
print("\nGenerazione mappe città-wide ...")

hex_map_plot(hex_map, "dens_tot",
             "Densità di sosta — totale (soste/km²)\nComune di Roma, marzo 2023",
             "YlOrRd", "map01_densita_totale.png",
             mun_gdf=mun, label="Soste / km²")

hex_map_plot(hex_map, "dens_day",
             "Densità di sosta — diurna 07:00–20:00 (soste/km²)\nComune di Roma, marzo 2023",
             "YlOrRd", "map02_densita_diurna.png",
             mun_gdf=mun, label="Soste / km² (07–20)")

hex_map_plot(hex_map, "dens_night",
             "Densità di sosta — notturna 20:00–07:00 (soste/km²)\nComune di Roma, marzo 2023",
             "Blues", "map03_densita_notturna.png",
             mun_gdf=mun, label="Soste / km² (20–07)")

# Indice residenziale: % sosta notturna (celle con ≥10 soste)
hex_night_idx = hex_map[hex_map["n_tot"] >= 10].copy()
hex_map_plot(hex_night_idx, "pct_night",
             "Indice residenziale — % soste notturne (20:00–07:00)\n"
             "Alta % = zona a prevalente sosta residenziale",
             "PuBu", "map04_indice_residenziale.png",
             mun_gdf=mun, label="% soste notturne", pct_clip=95)

# ── 9. Mappe per municipio ────────────────────────────────────────────────────
print("\nGenerazione mappe per municipio ...")

import matplotlib.patheffects as pe  # already imported above, harmless

choropleth(mun_stats, "dens_tot",
           "Densità di sosta per municipio — totale (soste/km²)",
           "YlOrRd", "map05_municipio_densita_totale.png",
           label="Soste / km²")

choropleth(mun_stats, "dens_day",
           "Densità di sosta per municipio — diurna 07:00–20:00 (soste/km²)",
           "YlOrRd", "map06_municipio_densita_diurna.png",
           label="Soste / km² (07–20)")

choropleth(mun_stats, "dens_night",
           "Densità di sosta per municipio — notturna 20:00–07:00 (soste/km²)",
           "Blues", "map07_municipio_densita_notturna.png",
           label="Soste / km² (20–07)")

choropleth(mun_stats, "pct_night",
           "Indice residenziale per municipio — % soste notturne",
           "PuBu", "map08_municipio_indice_residenziale.png",
           label="% soste notturne", fmt="{:.1f}%")

# ── 10. Bar chart indicatori per municipio ───────────────────────────────────
print("  Bar chart indicatori municipio ...")
ms = mun_stats.sort_values("dens_tot", ascending=True)
labels = [f"M{int(r.Numero)} – {r.Name[:18]}" for _, r in ms.iterrows()]

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for ax, col, title, color in [
    (axes[0], "dens_tot",   "Densità totale\n(soste/km²)", "#E53935"),
    (axes[1], "dens_day",   "Densità diurna\n07–20 (soste/km²)", "#FB8C00"),
    (axes[2], "pct_night",  "Indice residenziale\n(% soste notturne)", "#1565C0"),
]:
    ax.barh(labels, ms[col], color=color, alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(col.replace("_", " "))
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle("Indicatori di sosta per municipio — Roma, marzo 2023",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save_map(fig, "map09_municipio_barchart_indicatori.png")

# ── 11. Mappe AC_VEI (se disponibili) ────────────────────────────────────────
if has_onstreet:
    print("\nGenerazione mappe on-street / saturazione (AC_VEI) ...")

    choropleth(mun_stats, "pct_onstreet_2m",
               "% soste on-street per municipio (buffer 2 m)",
               "RdYlGn", "map10_municipio_pct_onstreet.png",
               label="% on-street (≤2m)", fmt="{:.1f}%",
               vmin=mun_stats["pct_onstreet_2m"].min(),
               vmax=mun_stats["pct_onstreet_2m"].max())

    # Saturazione: serve area carrabile per municipio → richiede AC_VEI.shp
    ac_vei_path = os.path.join(BASE, "AC_VEI.shp")
    if is_real_shapefile(ac_vei_path):
        print("  Caricamento AC_VEI per saturazione ...")
        ac_vei = gpd.read_file(ac_vei_path).to_crs(TARGET_CRS)
        mun_with_ac = gpd.sjoin(ac_vei[["geometry"]],
                                mun_tag[["Numero", "geometry"]],
                                how="inner", predicate="intersects")
        ac_area_by_mun = (
            ac_vei.loc[mun_with_ac.index]
            .assign(Numero=mun_with_ac["Numero"].values)
            .groupby("Numero")
            .apply(lambda g: g.geometry.area.sum())
            .rename("area_carrabile_mq")
        )
        mun_stats = mun_stats.join(ac_area_by_mun, on="Numero")
        mun_stats["area_occupata_mq"] = (mun_stats["n_onstreet_2m"] * AREA_VEI_MQ)
        mun_stats["saturazione_pct"]  = (
            mun_stats["area_occupata_mq"] /
            mun_stats["area_carrabile_mq"].replace(0, np.nan) * 100
        )
        choropleth(mun_stats, "saturazione_pct",
                   "Saturazione superficie carrabile per municipio\n"
                   "(area occupata da sosta on-street / area carrabile totale)",
                   "RdYlGn_r", "map11_municipio_saturazione.png",
                   label="% saturazione", fmt="{:.1f}%")
        mun_stats.drop(columns="geometry").to_csv(
            os.path.join(RESULTS, "11_statistiche_municipio.csv"), index=False)
    else:
        print("  AC_VEI.shp non disponibile — mappa saturazione saltata.")
else:
    print("\n  [Mappe on-street/saturazione saltate: on_street_2m non in dati]")

# ── Riepilogo ─────────────────────────────────────────────────────────────────
print(f"\nTutte le mappe salvate in: {MAPS}/")
print("\nIndicatori per municipio (top 5 densità totale):")
top5 = mun_stats.nlargest(5, "dens_tot")[
    ["Numero", "Name", "n_tot", "dens_tot", "pct_night"]]
print(top5.to_string(index=False))
