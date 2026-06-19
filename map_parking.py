"""
Mappe di densità e pressione di parcheggio — Comune di Roma
============================================================
Griglia esagonale H3 (resolution 9, edge ~174 m) per le mappe città-wide.
Produce mappe in scala campione (soste FCD raw) e in scala città (calibrata).

Outputs in results/maps/:
  map01–03  Densità totale/diurna/notturna (campione, soste/km²)
  map04–06  Densità totale/diurna/notturna (stima città)
  map07     Indice residenziale (% soste notturne)
  map08–09  Densità per municipio (campione + città)
  map10     Indice residenziale per municipio
  map11     Bar chart indicatori per municipio
  map12–13  % on-street + saturazione per municipio (se AC_VEI disponibile)
  map14     Saturazione per cella H3 (se AC_VEI disponibile)
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from shapely.geometry import Polygon
from shapely import wkt
import h3

warnings.filterwarnings("ignore")

# ── Contextily (OSM basemap) — opzionale, con fallback automatico ─────────────
try:
    import contextily as ctx
    # CartoDB Positron: sfondo chiaro, non compete con i colori dei dati
    BASEMAP_PROVIDER = ctx.providers.CartoDB.Positron
    HAS_CTX = True
except ImportError:
    HAS_CTX = False

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
MAPS    = os.path.join(RESULTS, "maps")
os.makedirs(MAPS, exist_ok=True)

TARGET_CRS = "EPSG:25833"
H3_RES     = 9          # edge ~174m, area ~0.105 km²
DAY_HOURS  = range(7, 20)
AREA_VEI_MQ = 12.5      # ingombro medio per veicolo (m²)

# Calibration
REF_PEAK_VEHICLES = 300_000
PEAK_WINDOW_START = 7
PEAK_WINDOW_END   = 9
FLEET_TOTAL_ROME  = 1_600_000

plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def _add_basemap(ax, fallback_color="#dcdcdc"):
    """Aggiunge OSM basemap; se i tile non sono raggiungibili usa sfondo grigio."""
    if HAS_CTX:
        try:
            ctx.add_basemap(ax, crs=TARGET_CRS,
                            source=BASEMAP_PROVIDER,
                            attribution_size=6)
            return
        except Exception:
            pass
    ax.set_facecolor(fallback_color)


# ── Helpers ───────────────────────────────────────────────────────────────────
def is_real_shapefile(path):
    if not os.path.exists(path): return False
    with open(path, "rb") as f: magic = f.read(4)
    return magic == b'\x00\x00\x27\x0a'


def h3_cell_to_polygon(cell):
    """H3 cell boundary → Shapely Polygon (WGS84 lon/lat order)."""
    boundary = h3.cell_to_boundary(cell)
    return Polygon([(lng, lat) for lat, lng in boundary])


def save_map(fig, name):
    path = os.path.join(MAPS, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def hex_map_plot(hexdf, col, title, cmap, fname,
                 mun_gdf=None, label="", pct_clip=97, min_n=2, vmax=None):
    data = hexdf[hexdf["n_tot"] >= min_n].copy()
    if vmax is None:
        vmax = np.nanpercentile(data[col].dropna().values, pct_clip)
    fig, ax = plt.subplots(figsize=(14, 13))
    # Celle esagonali semi-trasparenti così il basemap traspare sotto
    data.plot(column=col, cmap=cmap, vmin=0, vmax=vmax,
              edgecolor="none", linewidth=0, ax=ax, alpha=0.78, legend=False)
    if mun_gdf is not None:
        mun_gdf.boundary.plot(ax=ax, color="#333", linewidth=1.4)
        for _, r in mun_gdf.iterrows():
            ax.annotate(f"M{int(r['Numero'])}",
                        xy=(r.geometry.centroid.x, r.geometry.centroid.y),
                        ha="center", va="center", fontsize=7, color="#111",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    _add_basemap(ax)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.01)
    cbar.set_label(label, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_axis_off()
    save_map(fig, fname)


def choropleth(gdf, col, title, cmap, fname,
               label="", fmt="{:.0f}", vmin=None, vmax=None):
    vmin_ = vmin if vmin is not None else gdf[col].quantile(0.05)
    vmax_ = vmax if vmax is not None else gdf[col].quantile(0.95)
    fig, ax = plt.subplots(figsize=(12, 11))
    # Poligoni semi-trasparenti con bordo marcato per leggibilità
    gdf.plot(column=col, cmap=cmap, vmin=vmin_, vmax=vmax_,
             edgecolor="#333", linewidth=1.5, ax=ax, legend=False, alpha=0.70)
    for _, row in gdf.iterrows():
        c = row.geometry.centroid
        ax.annotate(f"M{int(row['Numero'])}\n{fmt.format(row[col])}",
                    xy=(c.x, c.y), ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground="#333")])
    _add_basemap(ax)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin_, vmax=vmax_))
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
trips["arr_time"]    = pd.to_datetime(trips["arr_time"])
trips["dep_time"]    = pd.to_datetime(trips["dep_time"])
trips["hour"]        = trips["arr_time"].dt.hour
trips["weekday_num"] = trips["arr_time"].dt.dayofweek
trips["geometry"]    = trips["d"].apply(wkt.loads)

gdf_dest = gpd.GeoDataFrame(trips, geometry="geometry", crs="EPSG:4326"
                             ).to_crs(TARGET_CRS)

mun = gpd.read_file(os.path.join(BASE, "municipi_2013.shp")).to_crs(TARGET_CRS)
mun_union_gdf = mun[["geometry"]].copy().reset_index(drop=True)

print("  Filtro Roma ...")
joined_roma = gpd.sjoin(gdf_dest.reset_index(), mun_union_gdf,
                        how="inner", predicate="within")
gdf_roma = gdf_dest.loc[joined_roma["index"].unique()].copy().reset_index(drop=True)
print(f"  Destinazioni dentro Roma: {len(gdf_roma):,}")

# ── 2. Calibrazione ───────────────────────────────────────────────────────────
print("\nCalibrazione campione → città ...")
peak = gdf_roma[
    (gdf_roma["dep_time"].dt.hour >= PEAK_WINDOW_START) &
    (gdf_roma["dep_time"].dt.hour <  PEAK_WINDOW_END)
]
avg_unique_peak = peak.groupby(peak["dep_time"].dt.date)["user_id"].nunique().mean()
SCALE = REF_PEAK_VEHICLES / avg_unique_peak
print(f"  Veicoli unici in punta 07–09 (campione): {avg_unique_peak:.0f}/giorno")
print(f"  Fattore di espansione: {SCALE:.1f}×")

# ── 3. Merge on/off-street ────────────────────────────────────────────────────
class_csv = os.path.join(RESULTS, "05_destinazioni_classificate.csv")
has_onstreet = False
pct_onstreet_global = 0.764   # fallback: valore a 2m dall'analisi precedente
if os.path.exists(class_csv):
    cls = pd.read_csv(class_csv, usecols=lambda c: c in ["trip_id", "on_street_2m"])
    if "on_street_2m" in cls.columns and "trip_id" in cls.columns:
        gdf_roma = gdf_roma.merge(cls, on="trip_id", how="left")
        has_onstreet = "on_street_2m" in gdf_roma.columns
        if has_onstreet:
            pct_onstreet_global = gdf_roma["on_street_2m"].mean()
            print(f"  % on-street 2m (campione): {pct_onstreet_global:.1%}")
print(f"  Dati on/off-street dettagliati: {has_onstreet}")

# ── 4. Split giorno / notte ───────────────────────────────────────────────────
is_day = gdf_roma["hour"].isin(DAY_HOURS)
gdf_day   = gdf_roma[is_day].copy()
gdf_night = gdf_roma[~is_day].copy()
print(f"  Sosta diurna: {len(gdf_day):,} | notturna: {len(gdf_night):,}")

# ── 5. H3 indexing ───────────────────────────────────────────────────────────
print("\nIndicizzazione H3 (resolution 9) ...")
gdf_wgs84 = gdf_roma.to_crs("EPSG:4326")
lats = gdf_wgs84.geometry.y.values
lons = gdf_wgs84.geometry.x.values
gdf_roma["h3_cell"] = [h3.latlng_to_cell(lat, lon, H3_RES)
                       for lat, lon in zip(lats, lons)]

# ── 6. Statistiche per cella H3 ───────────────────────────────────────────────
print("  Aggregazione per cella ...")
cell_agg = gdf_roma.groupby("h3_cell").agg(
    n_tot  =("h3_cell", "count"),
    n_day  =("hour",    lambda x: x.isin(DAY_HOURS).sum()),
).reset_index()
cell_agg["n_night"]  = cell_agg["n_tot"] - cell_agg["n_day"]
cell_agg["pct_night"] = np.where(
    cell_agg["n_tot"] > 0,
    cell_agg["n_night"] / cell_agg["n_tot"] * 100, np.nan)

if has_onstreet:
    os_per_cell = gdf_roma.groupby("h3_cell")["on_street_2m"].mean().rename("pct_onstreet")
    cell_agg = cell_agg.merge(os_per_cell.reset_index(), on="h3_cell", how="left")

# ── 7. H3 cells → GeoDataFrame ───────────────────────────────────────────────
print("  Creazione geometrie H3 ...")
polys = [h3_cell_to_polygon(c) for c in cell_agg["h3_cell"].values]
hex_gdf = gpd.GeoDataFrame(cell_agg, geometry=polys, crs="EPSG:4326"
                           ).to_crs(TARGET_CRS)
hex_gdf["area_km2"] = hex_gdf.geometry.area / 1e6

# Clip to Rome
roma_poly = mun.geometry.unary_union
hex_gdf = hex_gdf[hex_gdf.geometry.intersects(roma_poly)].copy()

# ── 8. Calcola densità campione e calibrata ───────────────────────────────────
hex_gdf["dens_tot_raw"]   = hex_gdf["n_tot"]   / hex_gdf["area_km2"]
hex_gdf["dens_day_raw"]   = hex_gdf["n_day"]   / hex_gdf["area_km2"]
hex_gdf["dens_night_raw"] = hex_gdf["n_night"] / hex_gdf["area_km2"]
hex_gdf["dens_tot_city"]  = hex_gdf["dens_tot_raw"]   * SCALE
hex_gdf["dens_day_city"]  = hex_gdf["dens_day_raw"]   * SCALE
hex_gdf["dens_night_city"]= hex_gdf["dens_night_raw"] * SCALE

# ── 9. Mappe H3 città-wide ────────────────────────────────────────────────────
print("\nGenerazione mappe H3 città-wide ...")

# Campione
hex_map_plot(hex_gdf, "dens_tot_raw",
             "Densità di sosta — totale (campione FCD, soste/km²)\n"
             "Comune di Roma · marzo 2023 · H3 resolution 9 (~174m)",
             "YlOrRd", "map01_densita_totale_campione.png",
             mun_gdf=mun, label="Soste / km² (campione)")

hex_map_plot(hex_gdf, "dens_day_raw",
             "Densità di sosta — diurna 07:00–20:00 (campione FCD, soste/km²)",
             "YlOrRd", "map02_densita_diurna_campione.png",
             mun_gdf=mun, label="Soste / km² 07–20 (campione)")

hex_map_plot(hex_gdf, "dens_night_raw",
             "Densità di sosta — notturna 20:00–07:00 (campione FCD, soste/km²)",
             "Blues", "map03_densita_notturna_campione.png",
             mun_gdf=mun, label="Soste / km² 20–07 (campione)")

# Stima città calibrata
hex_map_plot(hex_gdf, "dens_tot_city",
             f"Densità di sosta — totale (stima città ×{SCALE:.0f}, soste/km²)\n"
             "Comune di Roma · marzo 2023",
             "YlOrRd", "map04_densita_totale_citta.png",
             mun_gdf=mun, label=f"Soste / km² (stima città ×{SCALE:.0f})")

hex_map_plot(hex_gdf, "dens_day_city",
             f"Densità di sosta — diurna 07–20 (stima città ×{SCALE:.0f}, soste/km²)",
             "YlOrRd", "map05_densita_diurna_citta.png",
             mun_gdf=mun, label=f"Soste / km² 07–20 (stima città)")

hex_map_plot(hex_gdf, "dens_night_city",
             f"Densità di sosta — notturna 20–07 (stima città ×{SCALE:.0f}, soste/km²)",
             "Blues", "map06_densita_notturna_citta.png",
             mun_gdf=mun, label=f"Soste / km² 20–07 (stima città)")

# Indice residenziale (celle con ≥10 soste)
hex_map_plot(hex_gdf[hex_gdf["n_tot"] >= 10], "pct_night",
             "Indice residenziale — % soste notturne (20:00–07:00)\n"
             "Alta % = zona a prevalente sosta residenziale",
             "PuBu", "map07_indice_residenziale.png",
             mun_gdf=mun, label="% soste notturne", pct_clip=95)

# ── 10. Statistiche e mappe per municipio ─────────────────────────────────────
print("\nGenerazione mappe per municipio ...")
mun_tag = mun[["Numero", "Name", "geometry"]].copy().reset_index(drop=True)
gdf_tagged = gpd.sjoin(
    gdf_roma[["geometry", "hour"] + (["on_street_2m"] if has_onstreet else [])],
    mun_tag, how="left", predicate="within"
)

mun_stats = mun_tag.copy()
tot_by_mun   = gdf_tagged.groupby("Numero").size().rename("n_tot")
day_by_mun   = gdf_tagged[gdf_tagged["hour"].isin(DAY_HOURS)].groupby("Numero").size().rename("n_day")
night_by_mun = gdf_tagged[~gdf_tagged["hour"].isin(DAY_HOURS)].groupby("Numero").size().rename("n_night")

mun_stats = mun_stats.join(
    pd.DataFrame({"n_tot": tot_by_mun, "n_day": day_by_mun, "n_night": night_by_mun}),
    on="Numero"
).fillna(0)

mun_stats["area_km2"]      = mun_stats.geometry.area / 1e6
mun_stats["dens_tot"]      = mun_stats["n_tot"]   / mun_stats["area_km2"]
mun_stats["dens_day"]      = mun_stats["n_day"]   / mun_stats["area_km2"]
mun_stats["dens_night"]    = mun_stats["n_night"] / mun_stats["area_km2"]
mun_stats["pct_night"]     = (mun_stats["n_night"] /
                               mun_stats["n_tot"].replace(0, np.nan) * 100)
mun_stats["dens_tot_city"] = mun_stats["dens_tot"]   * SCALE
mun_stats["dens_day_city"] = mun_stats["dens_day"]   * SCALE
mun_stats["dens_night_city"]= mun_stats["dens_night"] * SCALE
mun_stats["n_tot_city"]    = (mun_stats["n_tot"] * SCALE).round(0).astype(int)

if has_onstreet:
    on2 = (gdf_tagged[gdf_tagged["on_street_2m"] == 1]
           .groupby("Numero").size().rename("n_onstreet_2m"))
    mun_stats = mun_stats.join(on2, on="Numero").fillna(0)
    mun_stats["pct_onstreet_2m"] = (mun_stats["n_onstreet_2m"] /
                                     mun_stats["n_tot"].replace(0, np.nan) * 100)

mun_stats.drop(columns="geometry").to_csv(
    os.path.join(RESULTS, "11_statistiche_municipio.csv"), index=False)

choropleth(mun_stats, "dens_tot",
           "Densità di sosta per municipio — totale, campione (soste/km²)",
           "YlOrRd", "map08_municipio_densita_campione.png",
           label="Soste / km² (campione)")

choropleth(mun_stats, "dens_tot_city",
           f"Densità di sosta per municipio — stima città (soste/km², ×{SCALE:.0f})",
           "YlOrRd", "map09_municipio_densita_citta.png",
           label=f"Soste / km² (stima città, ×{SCALE:.0f})")

choropleth(mun_stats, "pct_night",
           "Indice residenziale per municipio — % soste notturne (20:00–07:00)",
           "PuBu", "map10_municipio_indice_residenziale.png",
           label="% soste notturne", fmt="{:.1f}%")

if has_onstreet:
    choropleth(mun_stats, "pct_onstreet_2m",
               "% soste on-street per municipio (buffer 2 m)",
               "RdYlGn", "map11_municipio_pct_onstreet.png",
               label="% on-street (≤2m)", fmt="{:.1f}%",
               vmin=mun_stats["pct_onstreet_2m"].min(),
               vmax=mun_stats["pct_onstreet_2m"].max())

# Bar chart comparativo
ms = mun_stats.sort_values("dens_tot", ascending=True)
labels = [f"M{int(r.Numero)} – {r.Name[:18]}" for _, r in ms.iterrows()]

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
for ax, col, title, color in [
    (axes[0], "dens_tot",      "Densità totale\n(soste/km², campione)",           "#E53935"),
    (axes[1], "dens_tot_city", f"Densità totale\n(soste/km², stima ×{SCALE:.0f})","#C62828"),
    (axes[2], "pct_night",     "Indice residenziale\n(% soste notturne)",          "#1565C0"),
]:
    ax.barh(labels, ms[col], color=color, alpha=0.85)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

fig.suptitle("Indicatori di sosta per municipio — Roma, marzo 2023",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
save_map(fig, "map12_municipio_barchart.png")

# ── 11. Saturazione (richiede AC_VEI.shp) ────────────────────────────────────
ac_vei_path = os.path.join(BASE, "AC_VEI.shp")
if is_real_shapefile(ac_vei_path):
    print("\nCaricamento AC_VEI per saturazione ...")
    ac_vei = gpd.read_file(ac_vei_path).to_crs(TARGET_CRS)

    # Saturazione per municipio
    ac_in_mun = gpd.sjoin(ac_vei[["geometry"]], mun_tag[["Numero", "geometry"]],
                          how="inner", predicate="intersects")
    ac_area_mun = (
        ac_vei.loc[ac_in_mun.index]
        .assign(Numero=ac_in_mun["Numero"].values)
        .groupby("Numero")
        .apply(lambda g: g.geometry.area.sum())
        .rename("area_carrabile_mq")
    )
    mun_stats = mun_stats.join(ac_area_mun, on="Numero")
    n_on_col = "n_onstreet_2m" if has_onstreet else "n_tot"
    scale_col = mun_stats[n_on_col] if has_onstreet else mun_stats["n_tot"] * pct_onstreet_global
    mun_stats["surf_occ_mq"]    = scale_col * SCALE * AREA_VEI_MQ
    mun_stats["saturazione_pct"] = (
        mun_stats["surf_occ_mq"] /
        mun_stats["area_carrabile_mq"].replace(0, np.nan) * 100
    )
    choropleth(mun_stats, "saturazione_pct",
               "Saturazione superficie carrabile per municipio\n"
               "(soste on-street stimate × 12.5 m² / area AC_VEI)",
               "RdYlGn_r", "map13_municipio_saturazione.png",
               label="% saturazione", fmt="{:.1f}%")
    mun_stats.drop(columns="geometry").to_csv(
        os.path.join(RESULTS, "11_statistiche_municipio.csv"), index=False)

    # Saturazione per cella H3
    print("  Saturazione per cella H3 ...")
    ac_in_hex = gpd.sjoin(ac_vei[["geometry"]],
                          hex_gdf[["h3_cell", "geometry"]],
                          how="inner", predicate="intersects")
    ac_area_hex = (
        ac_vei.loc[ac_in_hex.index]
        .assign(h3_cell=ac_in_hex["h3_cell"].values)
        .groupby("h3_cell")
        .apply(lambda g: g.geometry.area.sum())
        .rename("area_carrabile_mq")
    )
    hex_gdf = hex_gdf.join(ac_area_hex, on="h3_cell")
    if has_onstreet:
        hex_gdf["n_on_city"] = hex_gdf["h3_cell"].map(
            gdf_roma.groupby("h3_cell")["on_street_2m"].sum() * SCALE
        ).fillna(0)
    else:
        hex_gdf["n_on_city"] = hex_gdf["n_tot"] * SCALE * pct_onstreet_global
    hex_gdf["saturazione_pct"] = (
        hex_gdf["n_on_city"] * AREA_VEI_MQ /
        hex_gdf["area_carrabile_mq"].replace(0, np.nan) * 100
    )
    hex_valid = hex_gdf[
        (hex_gdf["n_tot"] >= 3) & hex_gdf["area_carrabile_mq"].notna()
    ].copy()
    hex_map_plot(hex_valid, "saturazione_pct",
                 "Saturazione superficie carrabile per cella H3 (stima città)\n"
                 "soste on-street stimate × 12.5 m² / area AC_VEI nella cella",
                 "RdYlGn_r", "map14_saturazione_h3.png",
                 mun_gdf=mun, label="% saturazione", pct_clip=95)
else:
    print("\n  [Mappe saturazione saltate: AC_VEI.shp non disponibile in ambiente corrente]")

# ── Riepilogo ─────────────────────────────────────────────────────────────────
print(f"\nTutte le mappe salvate in: {MAPS}/")
print(f"\nTop 5 municipio per densità totale campione:")
top5 = mun_stats.nlargest(5, "dens_tot")[
    ["Numero", "Name", "n_tot", "n_tot_city", "dens_tot", "dens_tot_city", "pct_night"]]
print(top5.to_string(index=False))
