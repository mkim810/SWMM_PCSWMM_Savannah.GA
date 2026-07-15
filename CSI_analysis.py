"""
SAR Roughness vs HEC-RAS Spatial Inter-Comparison
══════════════════════════════════════════════════
CORRECT FRAMING:
  Neither SAR roughness nor HEC-RAS is ground truth.
  CSI = TP/(TP+FP+FN) measures SPATIAL AGREEMENT
  between two independent flood indicators.

  Indicator A : SAR water surface roughness (observational)
  Indicator B : HEC-RAS depth > threshold   (model)

  TP = both indicators agree: flooded
  FP = SAR only  (roughness detected, model did not predict)
  FN = model only (model predicted, SAR did not detect)

Permanent water mask:
  HEC-RAS pixels wet (depth >= 0.05m) during pre-hurricane
  base flow are excluded from both indicators.
  Isolates hurricane flood signal from background channel.

References:
  Otsu (1979) IEEE Trans. Syst. Man Cybern. 9(1):62-66
  Huang et al. (2019) ISPRS J. Photogramm. 159:53-62
  Tran et al. (2022) Remote Sensing 14(22):5721
  Mason et al. (2021) Remote Sensing 13(13):2493
  Valenzuela (1978) Radio Science 13(6):1002-1012
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping
from skimage.filters import threshold_otsu
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────────────────────────

PERM_WATER_PATH = (
    r'F:\Users\minja\mark1_doc\Urban_model'
    r'\Combined_Water_Mask_30m.tif'
)
ROUGHNESS_PATH = (
    r'F:/Users/minja/mark1_doc/Urban_model/small_coeff'
    r'/Merged_Result_V2_clipped.tiff'
)
DEPTH_PATH = (
    r'F:\Users\minja\mark1_doc\Urban_model\small_coeff\Depth (09OCT2016 23 45 00).Mod_RAS_Channel_Ter.Mod_RAS_DEM_Ch_Te.tif'
)
VELOCITY_PATH = (
    r'E:\RASModel\0915Home\Matthew_Long_MOD'
    r'\Velocity (09OCT2016 23 45 00)'
    r'.Mod_RAS_Channel_Ter.Mod_RAS_DEM_Ch_Te.tif'
)
SLOPE_PATH = (
    r'F:\Users\minja\mark1_doc\Urban_model'
    r'\slope_30m_clipped.tif'
)
DOMAIN_SHP = r'F:\Users\minja\mark1_doc\Urban_model\mod_fluvial_domain.shp'

V_THRESH  = 0.01   # m/s  fixed velocity threshold for comparison C
S_THRESH  = 0.01   # m/m  slope threshold (kept for reference only)

OUTPUT_CSV  = (r'F:\Users\minja\mark1_doc\Urban_model'
               r'\intercomparison_results_final_ref.csv')
OUTPUT_PREFIX = (r'F:\Users\minja\mark1_doc\Urban_model'
                  r'\sar_intercomp_final')


# ─────────────────────────────────────────────────────────────────
# LOAD HELPERS
# ─────────────────────────────────────────────────────────────────

def get_target(depth_path):
    """
    Depth raster defines the authoritative target grid.
    All other layers are reprojected to match this grid.
    """
    with rasterio.open(depth_path) as s:
        return dict(
            crs       = s.crs,
            transform = s.transform,
            shape     = (s.height, s.width),
            res       = s.res,
            bounds    = s.bounds,
            profile   = s.profile.copy()
        )


def reproject_to_target(src_path, target,
                         resampling=Resampling.average,
                         nodata_fill=-9999.0):
    """
    ALWAYS reprojects src_path to target grid.
    Shape equality does NOT guarantee spatial alignment.
    For binary masks use Resampling.nearest.
    For continuous data use Resampling.average.
    """
    with rasterio.open(src_path) as s:
        src_data   = s.read(1).astype(np.float32)
        src_nodata = s.nodata
        src_tr     = s.transform
        src_crs    = s.crs
        print(f"    src : shape={s.shape}  res={s.res}  "
              f"CRS={s.crs.to_epsg()}")
        print(f"    dst : shape={target['shape']}  "
              f"res={target['res']}  "
              f"CRS={target['crs'].to_epsg()}")

    if src_nodata is not None:
        src_data[src_data == src_nodata] = nodata_fill
    src_data[~np.isfinite(src_data)] = nodata_fill

    dst_data = np.full(target['shape'], nodata_fill, dtype=np.float32)
    reproject(
        source        = src_data,
        destination   = dst_data,
        src_transform = src_tr,
        src_crs       = src_crs,
        dst_transform = target['transform'],
        dst_crs       = target['crs'],
        resampling    = resampling,
        src_nodata    = nodata_fill,
        dst_nodata    = nodata_fill
    )

    out = dst_data.astype(float)
    out[out == nodata_fill] = np.nan
    out[~np.isfinite(out)]  = np.nan
    return out


def load_perm_water_mask(perm_water_path, target):
    """
    Load permanent water mask and resample to target grid.
    Permanent water = pixels wet during pre-hurricane base flow.
    Excluded from both SAR and HEC-RAS indicators.
    """
    print(f"\n  Loading permanent water mask...")
    arr = reproject_to_target(
        perm_water_path, target,
        resampling  = Resampling.nearest,
        nodata_fill = -9999.0
    )
    perm_water = np.where(np.isnan(arr), False, arr > 0.5)
    n_perm = perm_water.sum()
    print(f"    Permanent water pixels: {n_perm:,}")
    assert perm_water.shape == target['shape'], \
        "Shape mismatch after reproject"
    return perm_water


def load_hecras(path):
    """Load HEC-RAS raster directly."""
    with rasterio.open(path) as s:
        d      = s.read(1).astype(float)
        nodata = s.nodata
    if nodata is not None:
        d[d == nodata] = np.nan
    d[d < 0] = np.nan
    return d


def slope_load(path, target):
    """Load slope at native resolution, resample to target."""
    with rasterio.open(path) as s:
        raw    = s.read(1).astype(float)
        res    = s.res[0]
        nodata = s.nodata
        s_tr   = s.transform
        s_crs  = s.crs
    if nodata is not None:
        raw[raw == nodata] = np.nan
    if np.nanmax(raw) > 10:
        filled = np.where(np.isnan(raw), 0., raw)
        gx = np.gradient(filled, res, axis=1)
        gy = np.gradient(filled, res, axis=0)
        raw = np.sqrt(gx**2 + gy**2)
    f32 = raw.astype(np.float32)
    f32[np.isnan(raw)] = -9999.
    dst = np.full(target['shape'], -9999., dtype=np.float32)
    reproject(source=f32, destination=dst,
              src_transform=s_tr, src_crs=s_crs,
              dst_transform=target['transform'],
              dst_crs=target['crs'],
              resampling=Resampling.average,
              src_nodata=-9999., dst_nodata=-9999.)
    out = dst.astype(float)
    out[out == -9999.] = np.nan
    return out


def rasterize_shp(shp_path, target):
    gdf = gpd.read_file(shp_path)
    if gdf.crs != target['crs']:
        gdf = gdf.to_crs(target['crs'])
    geoms = [(mapping(g), 1) for g in gdf.geometry
             if g is not None and g.is_valid]
    burned = rasterize(geoms, out_shape=target['shape'],
                       transform=target['transform'],
                       fill=0, dtype=np.uint8, all_touched=False)
    return burned.astype(bool)


def normalize_sar(roughness_raw, valid):
    """Normalize SAR roughness to [0,1] within valid domain."""
    vals = roughness_raw[valid]
    lo   = np.percentile(vals, 2)
    hi   = np.percentile(vals, 98)
    norm = np.clip((roughness_raw - lo) / (hi - lo + 1e-9), 0, 1)
    norm[~valid] = np.nan
    return norm, lo, hi


# ─────────────────────────────────────────────────────────────────
# SPATIAL AGREEMENT METRICS
# ─────────────────────────────────────────────────────────────────

def compute_metrics(sar_indicator, hecras_indicator, mask,
                     label=""):
    """
    Compute spatial agreement between SAR and HEC-RAS indicators.

    INTER-COMPARISON FRAMING — neither is ground truth:
      sar_indicator   : SAR roughness-based flood indicator (A)
      hecras_indicator: HEC-RAS depth/velocity indicator   (B)

      TP = both A and B indicate flooding   (agreed flooded)
      FP = A only  — SAR detects, model does not predict
      FN = B only  — model predicts, SAR does not detect
      TN = both agree dry

    CSI = TP / (TP + FP + FN)
    """
    a  = sar_indicator[mask]
    b  = hecras_indicator[mask]

    tp  = int(( a &  b).sum())
    fp  = int(( a & ~b).sum())
    fn  = int((~a &  b).sum())
    tn  = int((~a & ~b).sum())
    n   = tp + fp + fn + tn + 1e-9

    csi   = tp / (tp + fp + fn + 1e-9)
    rec   = tp / (tp + fn + 1e-9)
    pre   = tp / (tp + fp + 1e-9)
    p_obs = (tp + tn) / n
    p_exp = (((tp+fp)/n)*((tp+fn)/n) + ((fn+tn)/n)*((fp+tn)/n))
    kappa = (p_obs - p_exp) / (1 - p_exp + 1e-9)

    if label:
        print(f"  {label:<28s}| "
              f"CSI={csi:.3f}  Rec={rec:.3f}  Pre={pre:.3f}  "
              f"κ={kappa:.3f}  "
              f"TP={tp:,}  FP={fp:,}  FN={fn:,}")

    return dict(csi=csi, recall=rec, precision=pre,
                kappa=kappa, tp=tp, fp=fp, fn=fn, tn=tn)


# ─────────────────────────────────────────────────────────────────
# THREE SWEEP FUNCTIONS
# Each sweeps its own threshold axis
# ─────────────────────────────────────────────────────────────────

def sweep_A_sar_vs_depth(sar_indicator, depth, valid):
    """
    Comparison A: SAR roughness vs HEC-RAS depth indicator.
    Sweep: depth threshold 0.01 → 2.0m
    HEC-RAS indicator B = depth > d_thresh
    """
    rows = []
    for d_thresh in np.arange(0.01, 2.01, 0.05):
        hecras_indicator = (depth > d_thresh) & valid
        m = compute_metrics(sar_indicator, hecras_indicator, valid)
        rows.append(dict(
            comparison  = 'SAR_vs_depth',
            threshold   = round(float(d_thresh), 3),
            thresh_axis = 'depth_m',
            **{k: round(v, 4) if isinstance(v, float)
               else v for k, v in m.items()}
        ))
    return pd.DataFrame(rows)


def sweep_B_sar_vs_velocity(sar_indicator, velocity, valid):
    """
    Comparison B: SAR roughness vs HEC-RAS velocity indicator.
    Sweep: velocity threshold 0.01 → 2.0 m/s
    HEC-RAS indicator B = velocity > v_thresh

    NOTE: velocity threshold is the loop variable.
    The velocity mask changes at every step.
    """
    rows = []
    for v_thresh in np.arange(0.01, 2.01, 0.05):
        hecras_indicator = (velocity > v_thresh) & valid
        if hecras_indicator.sum() == 0:
            break
        m = compute_metrics(sar_indicator, hecras_indicator, valid)
        rows.append(dict(
            comparison  = 'SAR_vs_velocity',
            threshold   = round(float(v_thresh), 3),
            thresh_axis = 'velocity_ms',
            **{k: round(v, 4) if isinstance(v, float)
               else v for k, v in m.items()}
        ))
    return pd.DataFrame(rows)


def sweep_C_sar_vs_depth_and_velocity(sar_indicator, depth,
                                       velocity, valid,
                                       v_thresh=V_THRESH):
    """
    Comparison C: SAR roughness vs HEC-RAS depth AND velocity.
    Sweep: depth threshold 0.01 → 2.0m (velocity fixed)
    HEC-RAS indicator B = depth > d_thresh AND velocity > v_thresh
    """
    vel_indicator = (velocity > v_thresh) & valid
    rows = []
    for d_thresh in np.arange(0.01, 2.01, 0.05):
        hecras_indicator = (depth > d_thresh) & vel_indicator
        if hecras_indicator.sum() == 0:
            continue
        m = compute_metrics(sar_indicator, hecras_indicator, valid)
        rows.append(dict(
            comparison  = 'SAR_vs_depth_AND_velocity',
            threshold   = round(float(d_thresh), 3),
            thresh_axis = f'depth_m_vfix{v_thresh}ms',
            **{k: round(v, 4) if isinstance(v, float)
               else v for k, v in m.items()}
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# GRID ALIGNMENT REPORT
# ─────────────────────────────────────────────────────────────────

def print_grid_report(target, arrays_dict):
    print(f"\n{'='*65}")
    print(f"GRID ALIGNMENT REPORT")
    print(f"{'='*65}")
    print(f"  {'Layer':<20s}  {'Shape':<18s}  Min       Max")
    print(f"  {'-'*60}")
    for name, arr in arrays_dict.items():
        a = arr.astype(float) if arr.dtype == bool else arr
        vmin = float(np.nanmin(a)) if np.isfinite(a).any() else float('nan')
        vmax = float(np.nanmax(a)) if np.isfinite(a).any() else float('nan')
        match = "✓" if arr.shape == target['shape'] else "✗ MISMATCH"
        print(f"  {name:<20s}  {str(arr.shape):<18s}  "
              f"{vmin:8.3f}  {vmax:8.3f}  {match}")
    shapes = set(a.shape for a in arrays_dict.values())
    if len(shapes) > 1:
        print(f"\n  ✗ SHAPES DIFFER — check reproject functions")
    else:
        print(f"\n  ✓ All layers aligned to {target['shape']}")

# Insert this above the "def run():" line
def save_binary_tif(array, target_profile, filename, mask):
    """Saves a boolean array as a GeoTIFF, applying the domain mask."""
    profile = target_profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, nodata=255)
    
    # Prepare data: 1=Flood, 0=Dry, 255=Excluded/NoData
    out_arr = np.full(array.shape, 255, dtype=np.uint8)
    out_arr[mask] = array[mask].astype(np.uint8)
    
    with rasterio.open(filename, 'w', **profile) as dst:
        dst.write(out_arr, 1)
    print(f"  Saved image: {filename}")
# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────

def run():
    print("=" * 70)
    print("SAR ROUGHNESS vs HEC-RAS SPATIAL INTER-COMPARISON")
    print("10m resolution | Hurricane Matthew | Oct 9 2016")
    print("=" * 70)
    print("\nFraming: INTER-COMPARISON — neither dataset is ground truth")
    print("CSI = TP/(TP+FP+FN) measures spatial agreement")
    print("  TP = both SAR and HEC-RAS indicate flooding")
    print("  FP = SAR only (roughness detected, model did not predict)")
    print("  FN = model only (model predicted, SAR did not detect)")

    # ── 1. Target grid ────────────────────────────────────────────
    print("\n[1/8] Target grid (authoritative = HEC-RAS depth):")
    target = get_target(DEPTH_PATH)
    print(f"  Shape : {target['shape']}")
    print(f"  Res   : {target['res']} m")
    print(f"  CRS   : {target['crs']}")

    # ── 2. HEC-RAS depth ─────────────────────────────────────────
    print("\n[2/8] Loading HEC-RAS depth...")
    depth = load_hecras(DEPTH_PATH)
    if depth.shape != target['shape']:
        print(f"  Reprojecting depth {depth.shape} → {target['shape']}")
        depth = reproject_to_target(DEPTH_PATH, target,
                                     Resampling.average)
    depth[depth < 0] = np.nan
    print(f"  Depth range: {np.nanmin(depth):.2f} – "
          f"{np.nanmax(depth):.2f} m")

    # ── 3. HEC-RAS velocity ───────────────────────────────────────
    print("\n[3/8] Loading HEC-RAS velocity...")
    velocity = load_hecras(VELOCITY_PATH)
    if velocity.shape != target['shape']:
        print(f"  Reprojecting velocity {velocity.shape} "
              f"→ {target['shape']}")
        velocity = reproject_to_target(VELOCITY_PATH, target,
                                        Resampling.average)
    velocity[velocity < 0] = np.nan
    vel_active = velocity[np.isfinite(velocity) & (velocity > 0)]
    print(f"  Velocity range (active): "
          f"{vel_active.min():.4f} – {vel_active.max():.4f} m/s")
    print(f"  Velocity P50={np.percentile(vel_active,50):.4f}  "
          f"P90={np.percentile(vel_active,90):.4f} m/s")

    # ── 4. Permanent water mask ───────────────────────────────────
    print("\n[4/8] Permanent water mask (pre-hurricane base flow):")
    perm_water = load_perm_water_mask(PERM_WATER_PATH, target)

    # ── 5. SAR roughness ──────────────────────────────────────────
    print("\n[5/8] SAR roughness (reproject to depth grid):")
    roughness_raw = reproject_to_target(ROUGHNESS_PATH, target,
                                         Resampling.average)

    # ── 6. Slope ──────────────────────────────────────────────────
    print("\n[6/8] Slope (native res → resample to depth grid):")
    slope = slope_load(SLOPE_PATH, target)

    # ── 7. Domain ─────────────────────────────────────────────────
    print("\n[7/8] Rasterizing domain shapefile...")
    domain = rasterize_shp(DOMAIN_SHP, target)

    # Grid alignment check
    print_grid_report(target, {
        'depth'      : depth,
        'velocity'   : velocity,
        'perm_water' : perm_water,
        'roughness'  : roughness_raw,
        'slope'      : slope,
        'domain'     : domain,
    })

    # Safety trim
    arrays = [depth, velocity, perm_water, roughness_raw, slope, domain]
    nr = min(a.shape[0] for a in arrays)
    nc = min(a.shape[1] for a in arrays)
    (depth, velocity, perm_water,
     roughness_raw, slope, domain) = (a[:nr, :nc] for a in arrays)

    # ── 8. Build valid mask and SAR indicator ─────────────────────
    print("\n[8/8] Building valid mask and running sweeps...")

    # Valid mask — permanent water EXCLUDED from both indicators
    valid = (domain &
             ~np.isnan(depth) &
             ~np.isnan(roughness_raw) &
             ~np.isnan(slope) &
             ~perm_water)          # ← permanent channel excluded

    # For before/after comparison — no permanent water exclusion
    valid_unmasked = (domain &
                      ~np.isnan(depth) &
                      ~np.isnan(roughness_raw) &
                      ~np.isnan(slope))

    n_domain  = domain.sum()
    n_perm    = (perm_water & domain).sum()
    n_valid   = valid.sum()
    pct_perm  = n_perm / max(n_domain, 1) * 100

    print(f"\n  Domain pixels         : {n_domain:,}")
    print(f"  Permanent water       : {n_perm:,} ({pct_perm:.1f}%)")
    print(f"  Valid for analysis    : {n_valid:,}")

    if pct_perm > 30:
        print(f"  ⚠ Perm water > 30% — verify base flow HEC-RAS files")
    elif pct_perm < 0.1:
        print(f"  ℹ Perm water < 0.1% — minimal effect on CSI expected")
    else:
        print(f"  ✓ Permanent water fraction physically reasonable")

    # ── SAR flood indicator (masked domain) ───────────────────────
    # Otsu threshold applied to normalized roughness
    # Ref: Otsu (1979); Tran et al. (2022); Huang et al. (2019)
    roughness_norm, lo, hi = normalize_sar(roughness_raw, valid)
    vals_clipped = np.clip(roughness_norm[valid],
                            np.percentile(roughness_norm[valid], 2),
                            np.percentile(roughness_norm[valid], 98))
    otsu_thresh  = threshold_otsu(vals_clipped)
    sar_indicator = (roughness_norm > otsu_thresh) & valid

    print(f"\n  SAR Otsu threshold    : {otsu_thresh:.4f}")
    print(f"  SAR flood pixels      : {sar_indicator.sum():,} "
          f"({sar_indicator.sum()/max(n_valid,1)*100:.1f}%)")

    # ── SAR flood indicator (unmasked — for comparison only) ──────
    # Use SAME normalization and threshold for fair comparison
    roughness_norm_u = np.clip(
        (roughness_raw - lo) / (hi - lo + 1e-9), 0, 1)
    roughness_norm_u[~valid_unmasked] = np.nan
    sar_indicator_u = (roughness_norm_u > otsu_thresh) & valid_unmasked

    # ── Three threshold sweeps ────────────────────────────────────
    print(f"\n  Running three threshold sweeps...")
    print(f"  (CSI = TP/(TP+FP+FN), no ground truth)")

    print(f"  A. SAR vs depth         (depth sweep 0.01–2.0m)...")
    df_A = sweep_A_sar_vs_depth(sar_indicator, depth, valid)

    print(f"  B. SAR vs velocity      (velocity sweep 0.01–2.0 m/s)...")
    df_B = sweep_B_sar_vs_velocity(sar_indicator, velocity, valid)

    print(f"  C. SAR vs depth+vel     "
          f"(depth sweep, vel>{V_THRESH}m/s fixed)...")
    df_C = sweep_C_sar_vs_depth_and_velocity(
        sar_indicator, depth, velocity, valid)

    # Verify velocity sweep is varying
    if df_B['csi'].nunique() == 1:
        print(f"  ⚠ WARNING: velocity CSI flat at "
              f"{df_B['csi'].iloc[0]:.4f} — check velocity raster")
    else:
        print(f"  ✓ Velocity sweep varies: "
              f"CSI {df_B['csi'].min():.3f}–{df_B['csi'].max():.3f}")

    # ── Unmasked comparison (A only) ──────────────────────────────
    rows_u = []
    for d_thresh in np.arange(0.01, 2.01, 0.05):
        hecras_u = (depth > d_thresh) & valid_unmasked
        if hecras_u.sum() == 0:
            continue
        m = compute_metrics(sar_indicator_u, hecras_u, valid_unmasked)
        rows_u.append(dict(
            comparison  = 'SAR_vs_depth_unmasked',
            threshold   = round(float(d_thresh), 3),
            thresh_axis = 'depth_m',
            **{k: round(v, 4) if isinstance(v, float)
               else v for k, v in m.items()}
        ))
    df_u = pd.DataFrame(rows_u)

    # Save combined CSV
    df_all = pd.concat([df_A, df_B, df_C, df_u], ignore_index=True)
    df_all.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  CSV saved: {OUTPUT_CSV}")

    # ── Best per comparison ───────────────────────────────────────
    best_A = df_A.loc[df_A['csi'].idxmax()]
    best_B = df_B.loc[df_B['csi'].idxmax()]
    best_C = df_C.loc[df_C['csi'].idxmax()]
    best_u = df_u.loc[df_u['csi'].idxmax()]

    sar_now  = best_A
    sar_orig = best_u
    delta    = sar_now['csi'] - sar_orig['csi']

    # ── Print results ─────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"INTER-COMPARISON RESULTS (permanent water excluded)")
    print(f"{'='*80}")
    print(f"  TP=agreed flooded | FP=SAR only | FN=model only")
    print(f"\n  {'Comparison':<32s}| Best thresh   CSI    "
          f"Recall  Precis  Kappa")
    print(f"  {'-'*72}")
    for b, unit in [
            (best_A, 'm depth'),
            (best_B, 'm/s vel'),
            (best_C, 'm depth')]:
        print(f"  {b['comparison']:<32s}| "
              f"{b['threshold']:>7.3f} {unit:<9s}  "
              f"{b['csi']:>6.3f}  "
              f"{b['recall']:>6.3f}  "
              f"{b['precision']:>6.3f}  "
              f"{b['kappa']:>6.3f}")

    # At key depth thresholds
    print(f"\n  At key depth thresholds:")
    print(f"  {'Comparison':<32s}| d_thresh   CSI    "
          f"Recall  Precis  TP       FP       FN")
    print(f"  {'-'*85}")
    for d_t in [0.76, 0.81, 1.00]:
        for df_sub in [df_A, df_C]:
            row = df_sub.iloc[
                (df_sub['threshold']-d_t).abs().argsort().iloc[0]]
            print(f"  {row['comparison']:<32s}| "
                  f"{d_t:>6.2f}m    "
                  f"{row['csi']:>6.3f}  "
                  f"{row['recall']:>6.3f}  "
                  f"{row['precision']:>6.3f}  "
                  f"{int(row['tp']):>7,}  "
                  f"{int(row['fp']):>7,}  "
                  f"{int(row['fn']):>7,}")

    # Velocity comparison at key thresholds
    print(f"\n  SAR vs velocity at key thresholds:")
    print(f"  {'v_thresh':<12s}| CSI    Recall  Precis  "
          f"TP       FP       FN")
    print(f"  {'-'*65}")
    for v_t in [0.01, 0.05, 0.10, 0.20, 0.30]:
        row = df_B.iloc[(df_B['threshold']-v_t).abs().argsort().iloc[0]]
        print(f"  {v_t:.2f} m/s    | "
              f"{row['csi']:>6.3f}  "
              f"{row['recall']:>6.3f}  "
              f"{row['precision']:>6.3f}  "
              f"{int(row['tp']):>7,}  "
              f"{int(row['fp']):>7,}  "
              f"{int(row['fn']):>7,}")

    # Before vs after masking
    print(f"\n{'='*65}")
    print(f"EFFECT OF PERMANENT WATER MASK (SAR vs depth, same norm)")
    print(f"{'='*65}")
    print(f"  Without mask : CSI={sar_orig['csi']:.3f}  "
          f"Rec={sar_orig['recall']:.3f}  "
          f"Pre={sar_orig['precision']:.3f}  "
          f"@ depth>{sar_orig['threshold']:.2f}m")
    print(f"  With mask    : CSI={sar_now['csi']:.3f}  "
          f"Rec={sar_now['recall']:.3f}  "
          f"Pre={sar_now['precision']:.3f}  "
          f"@ depth>{sar_now['threshold']:.2f}m")
    print(f"  ΔCSI         : {delta:+.3f}")
    if delta < -0.01:
        print(f"  → Channel pixels were inflating CSI via easy TPs")
        print(f"  → Masked CSI reflects genuine flood detection skill")
    elif abs(delta) <= 0.01:
        print(f"  → Negligible effect — channel pixels minor contributor")
    else:
        print(f"  → Unexpected increase — check mask")

    # ── TP/FP/FN arrays at best threshold for spatial map ─────────
    best_thresh_A = best_A['threshold']
    depth_flood   = (depth > best_thresh_A) & valid
    TP_map = sar_indicator  &  depth_flood
    FP_map = sar_indicator  & ~depth_flood
    FN_map = ~sar_indicator &  depth_flood
    TN_map = ~sar_indicator & ~depth_flood & valid

    print(f"\n  Spatial map arrays at depth>{best_thresh_A:.2f}m:")
    print(f"    TP={TP_map.sum():,}  FP={FP_map.sum():,}  "
          f"FN={FN_map.sum():,}  TN={TN_map.sum():,}")

    # ── Figures ───────────────────────────────────────────────────
    print(f"\nGenerating figures...")

    colors = {'A': '#2196F3', 'B': '#4CAF50', 'C': '#FF9800',
               'TP': '#2196F3', 'FP': '#FF9800', 'FN': '#F44336'}

    # ── Fig 1: CSI curves ─────────────────────────────────────────
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))
    fig1.suptitle(
        'SAR Roughness vs HEC-RAS: Spatial Agreement (CSI)\n'
        'Hurricane Matthew | Oct 9 2016 | 10m | '
        'Permanent water excluded',
        fontsize=12, fontweight='bold')

    ax1a.plot(df_A['threshold'], df_A['csi'],
               color=colors['A'], lw=2,
               label='A: SAR vs Depth')
    ax1a.plot(df_C['threshold'], df_C['csi'],
               color=colors['C'], ls=':', lw=2,
               label=f'C: SAR vs Depth+Vel (vel>{V_THRESH}m/s)')
    ax1a.plot(df_u['threshold'], df_u['csi'],
               color='gray', ls='--', lw=1.5, alpha=0.6,
               label='A: SAR vs Depth (no mask)')
    for t in [0.76, 0.81]:
        ax1a.axvline(t, color='gray', ls=':', lw=1, alpha=0.5)
    ax1a.axvline(best_thresh_A, color=colors['A'],
                  ls='--', lw=1.5, alpha=0.7,
                  label=f'Best d>{best_thresh_A:.2f}m '
                        f'CSI={best_A["csi"]:.3f}')
    ax1a.set_xlabel('HEC-RAS Depth Threshold (m)', fontsize=12)
    ax1a.set_ylabel('Spatial Agreement (CSI)', fontsize=12)
    ax1a.set_title('A and C: Depth Threshold Sweep',
                    fontsize=12, fontweight='bold')
    ax1a.set_xlim(0.01, 2.0); ax1a.set_ylim(0, 1.0)
    ax1a.legend(fontsize=9); ax1a.grid(True, alpha=0.3)
    ax1a.tick_params(labelsize=11)

    ax1b.plot(df_B['threshold'], df_B['csi'],
               color=colors['B'], lw=2,
               label='B: SAR vs Velocity')
    ax1b.axvline(best_B['threshold'], color=colors['B'],
                  ls='--', lw=1.5, alpha=0.7,
                  label=f"Best v>{best_B['threshold']:.2f}m/s "
                        f"CSI={best_B['csi']:.3f}")
    ax1b.axvline(V_THRESH, color='gray', ls=':', lw=1, alpha=0.5,
                  label=f'v={V_THRESH}m/s (ref)')
    ax1b.axvline(0.10, color='orange', ls=':', lw=1, alpha=0.5,
                  label='v=0.10 m/s (Bragg)')
    ax1b.set_xlabel('HEC-RAS Velocity Threshold (m/s)', fontsize=12)
    ax1b.set_ylabel('Spatial Agreement (CSI)', fontsize=12)
    ax1b.set_title('B: Velocity Threshold Sweep',
                    fontsize=12, fontweight='bold')
    ax1b.set_xlim(0.01, 2.0); ax1b.set_ylim(0, 1.0)
    ax1b.legend(fontsize=9); ax1b.grid(True, alpha=0.3)
    ax1b.tick_params(labelsize=11)
    fig1.tight_layout()
    path1 = OUTPUT_PREFIX + '_fig1_csi_curves.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig1)
    print(f"  Saved: {path1}")

    # ── Fig 2: Precision-Recall space ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    styles = [
        (df_A, 'A: SAR vs Depth',         colors['A'], '-'),
        (df_B, 'B: SAR vs Velocity',       colors['B'], '--'),
        (df_C, 'C: SAR vs Depth+Velocity', colors['C'], ':'),
    ]
    all_rec, all_pre = [], []
    for df_s, label, color, ls in styles:
        ax2.scatter(df_s['recall']*100, df_s['precision']*100,
                     c=[color]*len(df_s), s=8, alpha=0.4)
        idx  = df_s['csi'].idxmax()
        best = df_s.loc[idx]
        ax2.scatter(best['recall']*100, best['precision']*100,
                     c=[color], s=100, marker='o',
                     edgecolors='k', lw=0.5, zorder=5,
                     label=f'{label}  CSI={best["csi"]:.3f}')
        all_rec.extend(df_s['recall'].values*100)
        all_pre.extend(df_s['precision'].values*100)

    x_min = max(0,   min(all_rec)-3)
    x_max = min(100, max(all_rec)+3)
    y_min = max(0,   min(all_pre)-3)
    y_max = min(100, max(all_pre)+3)
    for csi_level in [0.3, 0.4, 0.5, 0.6, 0.7]:
        r  = np.linspace(x_min/100, x_max/100, 300)
        p  = (csi_level*r) / (r - csi_level*r + 1e-9)
        ok = (p >= y_min/100) & (p <= y_max/100)
        if ok.sum() < 2:
            continue
        ax2.plot(r[ok]*100, p[ok]*100, 'k--', alpha=0.2, lw=0.8)
        ax2.text(r[ok][-1]*100-0.5, p[ok][-1]*100,
                  f'{csi_level}', fontsize=8, color='gray',
                  ha='right', va='bottom', clip_on=True)

    ax2.set_xlabel('Recall (%)', fontsize=13)
    ax2.set_ylabel('Precision (%)', fontsize=13)
    ax2.set_title('Precision-Recall Space\n'
                   '(★ = best CSI per comparison)',
                   fontsize=13, fontweight='bold')
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3); ax2.tick_params(labelsize=12)
    fig2.tight_layout()
    path2 = OUTPUT_PREFIX + '_fig2_precrecall.png'
    fig2.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig2)
    print(f"  Saved: {path2}")

    # ── Fig 3: Spatial TP/FP/FN map ───────────────────────────────
    import matplotlib.patches as mpatches
    rgb = np.ones((*TP_map.shape, 3), dtype=float)
    rgb[TN_map, :]  = [0.90, 0.90, 0.90]   # grey  — both dry
    rgb[TP_map, :]  = [0.13, 0.59, 0.95]   # blue  — agreed flooded
    rgb[FP_map, :]  = [1.00, 0.60, 0.00]   # orange — SAR only
    rgb[FN_map, :]  = [0.96, 0.26, 0.21]   # red   — model only
    rgb[~valid, :]  = [1.00, 1.00, 1.00]   # white — outside domain

    fig3, ax3 = plt.subplots(figsize=(5, 6))
    ax3.imshow(rgb, interpolation='nearest')
    patches = [
        mpatches.Patch(color=[0.13,0.59,0.95],
                        label=f'TP n={TP_map.sum():,}'),
        mpatches.Patch(color=[1.00,0.60,0.00],
                        label=f'FP (SAR only) n={FP_map.sum():,}'),
        mpatches.Patch(color=[0.96,0.26,0.21],
                        label=f'FN (model only) n={FN_map.sum():,}'),
        mpatches.Patch(color=[0.90,0.90,0.90],
                        label='TN (both dry)'),
    ]
    ax3.legend(handles=patches, fontsize=8, loc='lower left',
                framealpha=0.9)
    csi_at_best = TP_map.sum()/(TP_map.sum()+FP_map.sum()+FN_map.sum()+1e-9)
    ax3.text(0.98, 0.98,
              f'depth > {best_thresh_A:.2f}m\n'
              f'CSI = {csi_at_best:.3f}',
              transform=ax3.transAxes, ha='right', va='top',
              fontsize=9, fontweight='bold',
              bbox=dict(boxstyle='round', fc='white',
                        ec='gray', alpha=0.9))
    ax3.set_title(f'Spatial Agreement: SAR vs HEC-RAS\n'
                   f'TP / FP / FN at depth > {best_thresh_A:.2f}m',
                   fontsize=13, fontweight='bold')
    ax3.axis('off')
    fig3.tight_layout()
    path3 = OUTPUT_PREFIX + '_fig3_spatial_tpfpfn.png'
    fig3.savefig(path3, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig3)
    print(f"  Saved: {path3}")

    # ── Fig 4: FN breakdown pie ───────────────────────────────────
    V_THRESH_QUAD = 0.10
    S_THRESH_QUAD = 0.05
    fn_ff = (FN_map & (slope <  S_THRESH_QUAD)
              & (velocity >= V_THRESH_QUAD)).sum()
    fn_fs = (FN_map & (slope <  S_THRESH_QUAD)
              & (velocity <  V_THRESH_QUAD)).sum()
    fn_sf = (FN_map & (slope >= S_THRESH_QUAD)
              & (velocity >= V_THRESH_QUAD)).sum()
    fn_ss = (FN_map & (slope >= S_THRESH_QUAD)
              & (velocity <  V_THRESH_QUAD)).sum()

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    wedges, texts, autos = ax4.pie(
        [fn_ff, fn_fs, fn_sf, fn_ss],
        labels=['Flat+fast', 'Flat+slow',
                 'Steep+fast', 'Steep+slow'],
        colors=['#4CAF50','#F44336','#9C27B0','#FF9800'],
        autopct='%1.1f%%', startangle=90,
        pctdistance=0.75, textprops={'fontsize': 9})
    for a in autos:
        a.set_fontweight('bold'); a.set_fontsize(10)
    ax4.set_title(f'Model-Only (FN) Pixel Breakdown\n'
                   f'depth > {best_thresh_A:.2f}m  '
                   f'N={FN_map.sum():,}',
                   fontsize=13, fontweight='bold')
    fig4.tight_layout()
    path4 = OUTPUT_PREFIX + '_fig4_fn_breakdown.png'
    fig4.savefig(path4, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig4)
    print(f"  Saved: {path4}")

    # ── Fig 5: Agreement bar ──────────────────────────────────────
    row_A = df_A.iloc[(df_A['threshold']-best_thresh_A
                        ).abs().argsort().iloc[0]]
    row_B = df_B.iloc[(df_B['threshold']-0.01
                        ).abs().argsort().iloc[0]]
    row_C = df_C.iloc[(df_C['threshold']-best_thresh_A
                        ).abs().argsort().iloc[0]]

    comps   = ['A: SAR vs\nDepth',
                'B: SAR vs\nVelocity',
                'C: SAR vs\nDepth+Vel']
    tp_vals = [row_A['tp'], row_B['tp'], row_C['tp']]
    fp_vals = [row_A['fp'], row_B['fp'], row_C['fp']]
    fn_vals = [row_A['fn'], row_B['fn'], row_C['fn']]

    fig5, ax5 = plt.subplots(figsize=(5, 5))
    x = np.arange(len(comps)); w = 0.55
    ax5.bar(x, tp_vals, w, label='TP (agreed flooded)',
             color=colors['TP'], alpha=0.85,
             edgecolor='black', lw=0.7)
    ax5.bar(x, fp_vals, w, bottom=tp_vals,
             label='FP (SAR only)',
             color=colors['FP'], alpha=0.85,
             edgecolor='black', lw=0.7)
    ax5.bar(x, fn_vals, w,
             bottom=[t+p for t,p in zip(tp_vals, fp_vals)],
             label='FN (model only)',
             color=colors['FN'], alpha=0.85,
             edgecolor='black', lw=0.7)
    for i, (tp, fp, fn) in enumerate(
            zip(tp_vals, fp_vals, fn_vals)):
        csi_v = tp/(tp+fp+fn+1e-9)
        ax5.text(i, (tp+fp+fn)*1.02, f'CSI={csi_v:.3f}',
                  ha='center', va='bottom', fontsize=9,
                  fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(comps, fontsize=11)
    ax5.set_ylabel('Pixel count', fontsize=13)
    ax5.set_title('Pixel Agreement Breakdown\n'
                   f'A,C: depth>{best_thresh_A:.2f}m  '
                   f'B: vel>0.01m/s',
                   fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9, loc='upper right', ncol=1)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.tick_params(labelsize=11)
    ax5.set_ylim(0, max(t+p+f for t,p,f
                         in zip(tp_vals,fp_vals,fn_vals)) * 1.3)
    fig5.tight_layout()
    path5 = OUTPUT_PREFIX + '_fig5_agreement_bar.png'
    fig5.savefig(path5, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig5)
    print(f"  Saved: {path5}")

    # ── Fig 6: SAR detection rate — Slope × Velocity space ──────
    # Motivated by FN pie: Flat+fast = dominant miss category
    # Slope (x-axis) captures terrain flatness driver
    # Velocity (y-axis) captures flow energy driver
    # Together explain WHERE SAR misses occur physically

    slope_valid   = slope[valid & np.isfinite(slope)]
    vel_valid_all = velocity[valid & np.isfinite(velocity)
                              & np.isfinite(slope)]

    s_max  = np.percentile(slope_valid, 98)
    v_max  = np.percentile(vel_valid_all, 98)
    n_bins = 20

    s_edges = np.linspace(0, s_max, n_bins + 1)
    v_edges = np.linspace(0, v_max, n_bins + 1)
    det_rate  = np.full((n_bins, n_bins), np.nan)
    count_map = np.zeros((n_bins, n_bins), dtype=int)

    slope_fin = np.isfinite(slope)
    vel_fin   = np.isfinite(velocity)

    for i in range(n_bins):        # slope axis
        for j in range(n_bins):    # velocity axis
            mask_bin = (valid &
                        slope_fin &
                        vel_fin &
                        (slope    >= s_edges[i]) &
                        (slope    <  s_edges[i+1]) &
                        (velocity >= v_edges[j]) &
                        (velocity <  v_edges[j+1]))
            n = mask_bin.sum()
            count_map[j, i] = n
            if n >= 5:
                det_rate[j, i] = (sar_indicator[mask_bin].sum()
                                   / n * 100)

    # Annotate FN quadrant boundaries
    s_thresh_q = S_THRESH_QUAD   # 0.05 m/m — flat vs steep
    v_thresh_q = 0.10            # m/s  — slow vs fast (Bragg min)

    fig6, ax6 = plt.subplots(figsize=(6, 5))
    im = ax6.imshow(det_rate,
                     origin='lower', aspect='auto',
                     extent=[0, s_max, 0, v_max],
                     cmap='RdYlGn', vmin=0, vmax=100,
                     interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax6, label='SAR Detection Rate (%)')
    cbar.ax.tick_params(labelsize=10)

    # FN quadrant boundary lines
    ax6.axvline(s_thresh_q, color='white', ls='--', lw=1.5,
                 alpha=0.8, label=f'slope={s_thresh_q} m/m')
    ax6.axhline(v_thresh_q, color='white', ls=':',  lw=1.5,
                 alpha=0.8, label=f'vel={v_thresh_q} m/s (Bragg)')

    # Quadrant labels (match FN pie categories)
    pad_x = s_max * 0.03
    pad_y = v_max * 0.03
    mid_flat   = s_thresh_q / 2
    mid_steep  = (s_thresh_q + s_max) / 2
    mid_slow   = v_thresh_q / 2
    mid_fast   = (v_thresh_q + v_max) / 2

    for txt, sx, vy in [
            ('Flat+slow',   mid_flat,  mid_slow),
            ('Flat+fast',   mid_flat,  mid_fast),
            ('Steep+slow',  mid_steep, mid_slow),
            ('Steep+fast',  mid_steep, mid_fast),
    ]:
        ax6.text(sx, vy, txt, ha='center', va='center',
                  fontsize=8, color='white', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2',
                             fc='black', alpha=0.35, lw=0))

    ax6.set_xlabel('Terrain Slope (m/m)', fontsize=13)
    ax6.set_ylabel('HEC-RAS Velocity (m/s)', fontsize=13)
    ax6.set_title('SAR Detection Rate (%)\n'
                   'Slope × Velocity Space  |  '
                   'Green=detected  Red=missed',
                   fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9, loc='upper right',
                framealpha=0.7)
    ax6.tick_params(labelsize=12)

    fig6.tight_layout()
    path6 = OUTPUT_PREFIX + '_fig6_detection_rate_slope_vel.png'
    fig6.savefig(path6, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show(); plt.close(fig6)
    print(f"  Saved: {path6}")

    # Print quadrant detection rate summary
    print(f"\n  SAR detection rates by FN quadrant "
          f"(slope threshold={s_thresh_q}, "
          f"vel threshold={v_thresh_q} m/s):")
    flat   = slope   <  s_thresh_q
    steep  = slope   >= s_thresh_q
    slow   = velocity <  v_thresh_q
    fast   = velocity >= v_thresh_q
    fn_fin = np.isfinite(slope) & np.isfinite(velocity)
    for q_label, mask_q in [
            ('Flat+slow  (calm shallow)',
              valid & fn_fin & flat  & slow),
            ('Flat+fast  (unexpected miss)',
              valid & fn_fin & flat  & fast),
            ('Steep+slow (shadow/layover)',
              valid & fn_fin & steep & slow),
            ('Steep+fast (geometry+energy)',
              valid & fn_fin & steep & fast),
    ]:
        n_q      = mask_q.sum()
        n_detect = sar_indicator[mask_q].sum()
        rate     = n_detect / max(n_q, 1) * 100
        n_fn_q   = FN_map[mask_q].sum()
        print(f"    {q_label:<32s}: "
              f"n={n_q:>7,}  "
              f"SAR detect={rate:>5.1f}%  "
              f"FN={n_fn_q:>6,}")

    # ── Final paper summary ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PAPER SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Framing   : spatial inter-comparison (no ground truth)")
    print(f"  Metric    : CSI = TP/(TP+FP+FN)")
    print(f"  Perm mask : {n_perm:,} px ({pct_perm:.1f}%) excluded")
    print(f"  ΔCSI mask : {delta:+.3f}")
    print(f"\n  A. SAR vs depth:")
    print(f"     CSI={best_A['csi']:.3f}  Rec={best_A['recall']:.3f}  "
          f"Pre={best_A['precision']:.3f}  "
          f"@ depth>{best_A['threshold']:.2f}m")
    print(f"\n  B. SAR vs velocity:")
    print(f"     CSI={best_B['csi']:.3f}  Rec={best_B['recall']:.3f}  "
          f"Pre={best_B['precision']:.3f}  "
          f"@ vel>{best_B['threshold']:.3f}m/s")
    print(f"\n  C. SAR vs depth+velocity:")
    print(f"     CSI={best_C['csi']:.3f}  Rec={best_C['recall']:.3f}  "
          f"Pre={best_C['precision']:.3f}  "
          f"@ depth>{best_C['threshold']:.2f}m")

    return (df_all, df_A, df_B, df_C,
            sar_indicator, depth_flood, TP_map, FP_map, FN_map,
            best_A, best_B, best_C)


if __name__ == "__main__":
    results = run()
    
