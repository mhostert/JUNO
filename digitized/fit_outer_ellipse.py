#!/usr/bin/env python
"""Fit an ellipse to the outer envelope of a digitized point cloud.

A contour digitized by colour selection comes out as a band of finite width
rather than a curve, so "the ellipse" is ambiguous.  Three fits are provided:

* ``mvee``   -- the minimum-volume enclosing ellipse (Khachiyan).  This is the
                outer envelope in the strict sense: the smallest ellipse that
                contains every point.  Use this one for "the outer ellipse".
* ``hull``   -- a direct least-squares conic fit (Fitzgibbon et al.) to the
                convex-hull vertices.  Also tracks the outside of the band, but
                averages over hull vertices instead of touching every extreme
                point, so a single stray pixel moves it much less.
* ``centre`` -- a direct least-squares conic fit to all points, i.e. the middle
                of the band.  Shown for reference.

All fits are done in a whitened frame and mapped back, because the two axes of
these plots differ in scale by orders of magnitude.

Several contours can be fitted at once; the sigma level is taken from each
filename.  Fitted parameters are printed, checked against each other for the
expected sqrt(Delta chi^2) scaling, and written to a JSON file for re-use.

Usage
-----
    python digitized/fit_outer_ellipse.py \
        digitized/JUNO_60days_{1,2,3}sigma_solar.dat \
        --xlabel '$\\Delta m^2_{21}$ [$10^{-5}$ eV$^2$]' \
        --ylabel '$\\sin^2\\theta_{12}$' \
        --reference 7.50,0.3092,0.12,0.0087 \
        --out digitized/JUNO_60days_solar_ellipses

Outputs ``<out>.json`` (parameters), ``<out>_Nsigma.dat`` (polylines) and
``<out>.pdf``.  Load the JSON with ``reactor.plotting.load_ellipse_set``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Ellipse representation
# ---------------------------------------------------------------------------
@dataclass
class Ellipse:
    """Ellipse as (x - c)^T A (x - c) = 1."""

    A: np.ndarray
    center: np.ndarray

    @property
    def axes(self) -> np.ndarray:
        """Semi-axis lengths, descending."""

        vals = np.linalg.eigvalsh(self.A)
        return np.sort(1.0 / np.sqrt(np.maximum(vals, 1e-300)))[::-1]

    @property
    def angle_deg(self) -> float:
        """Orientation of the major axis, degrees counter-clockwise from +x."""

        vals, vecs = np.linalg.eigh(self.A)
        major = vecs[:, int(np.argmin(vals))]
        return float(np.degrees(np.arctan2(major[1], major[0])))

    @property
    def covariance(self) -> np.ndarray:
        """Covariance whose 1-sigma (Delta chi^2 = 1) contour is this ellipse."""

        return np.linalg.inv(self.A)

    def polyline(self, n: int = 721) -> np.ndarray:
        t = np.linspace(0.0, 2.0 * np.pi, n)
        circle = np.stack([np.cos(t), np.sin(t)])
        vals, vecs = np.linalg.eigh(self.A)
        transform = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-300)))
        return (transform @ circle).T + self.center

    def contains(self, points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        delta = np.asarray(points, float) - self.center
        return np.einsum("ij,jk,ik->i", delta, self.A, delta) <= 1.0 + tol

    def scaled(self, factor: float) -> "Ellipse":
        return Ellipse(self.A / factor**2, self.center)

    def transformed(self, shift: np.ndarray, scale: np.ndarray) -> "Ellipse":
        """Map from a whitened frame back to data units: x_data = x*scale + shift."""

        S = np.diag(1.0 / scale)
        return Ellipse(S @ self.A @ S, self.center * scale + shift)


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------
def mvee(points: np.ndarray, tol: float = 1e-10, max_iter: int = 100_000) -> Ellipse:
    """Minimum-volume enclosing ellipse, by Khachiyan's algorithm."""

    P = np.asarray(points, dtype=float)
    n, d = P.shape
    Q = np.vstack([P.T, np.ones(n)])
    u = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        X = Q @ (u[:, None] * Q.T)
        M = np.einsum("ji,jk,ki->i", Q, np.linalg.inv(X), Q)
        j = int(np.argmax(M))
        step = (M[j] - d - 1.0) / ((d + 1.0) * (M[j] - 1.0))
        if step <= tol:
            break
        u *= 1.0 - step
        u[j] += step

    c = P.T @ u
    A = np.linalg.inv(P.T @ (u[:, None] * P) - np.outer(c, c)) / d

    # Khachiyan converges only to within `tol`, so a few extreme points can sit
    # marginally outside.  Inflate by the smallest factor that encloses them all,
    # which restores the defining property exactly at negligible cost in area.
    delta = P - c
    r_max = np.einsum("ij,jk,ik->i", delta, A, delta).max()
    if r_max > 1.0:
        A = A / r_max
    return Ellipse(A, c)


def fit_conic(points: np.ndarray) -> Ellipse:
    """Direct least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1999)."""

    P = np.asarray(points, dtype=float)
    x, y = P[:, 0], P[:, 1]
    D = np.stack([x**2, x * y, y**2, x, y, np.ones_like(x)], axis=1)
    S = D.T @ D

    # constraint matrix for 4ac - b^2 = 1, which forces an elliptical solution
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2.0
    C[1, 1] = -1.0

    from scipy.linalg import eig

    vals, vecs = eig(S, C)
    # keep the finite eigenvalue whose eigenvector really is an ellipse
    best = None
    for k in range(vecs.shape[1]):
        if not np.isfinite(vals[k]):
            continue
        a, b, cc = vecs[:3, k].real
        disc = 4.0 * a * cc - b**2
        if disc <= 0:
            continue
        if best is None or vals[k].real > vals[best].real:
            best = k
    if best is None:
        raise ValueError("conic fit did not return an ellipse")

    a, b, cc, dd, e, f = vecs[:, best].real

    # convert a x^2 + b xy + c y^2 + d x + e y + f = 0 to (x - u)^T A (x - u) = 1
    M = np.array([[a, b / 2.0], [b / 2.0, cc]])
    v = np.array([dd / 2.0, e / 2.0])
    center = -np.linalg.solve(M, v)
    scale = center @ M @ center - f
    # the conic is defined only up to an overall sign; pick the one that makes
    # A positive definite
    if scale < 0:
        M, scale = -M, -scale
    if scale <= 0 or np.linalg.eigvalsh(M).min() <= 0:
        raise ValueError("conic fit did not return an ellipse")
    return Ellipse(M / scale, center)


def outer_ellipse(
    points: np.ndarray,
    method: str = "mvee",
    trim: float = 0.0,
) -> Ellipse:
    """Fit an ellipse to the outside of a point cloud, in a whitened frame.

    ``trim`` optionally drops that fraction of the most extreme points (in the
    current fitted metric) before refitting, which protects the MVEE against
    stray digitizer pixels.
    """

    P = np.asarray(points, dtype=float)
    shift = P.mean(axis=0)
    scale = P.std(axis=0)
    W = (P - shift) / scale

    keep = np.ones(len(W), dtype=bool)
    for _ in range(6):
        sub = W[keep]
        if method == "mvee":
            fit = mvee(sub)
        elif method == "hull":
            from scipy.spatial import ConvexHull

            fit = fit_conic(sub[ConvexHull(sub).vertices])
        elif method == "centre":
            fit = fit_conic(sub)
        else:
            raise ValueError(f"unknown method {method!r}")

        if trim <= 0.0:
            break
        delta = W - fit.center
        radius = np.einsum("ij,jk,ik->i", delta, fit.A, delta)
        cutoff = np.quantile(radius, 1.0 - trim)
        new_keep = radius <= cutoff
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep

    return fit.transformed(shift, scale)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
DELTA_CHI2_2DOF = {1: 2.30, 2: 6.18, 3: 11.83}


def describe(name: str, ell: Ellipse, points: np.ndarray, level: int | None) -> str:
    axes = ell.axes
    inside = ell.contains(points).sum()
    lines = [
        f"{name}",
        f"  centre            : ({ell.center[0]:.5g}, {ell.center[1]:.6g})",
        f"  semi-axes         : {axes[0]:.5g}  x  {axes[1]:.5g}",
        f"  major-axis angle  : {ell.angle_deg:+.2f} deg",
        f"  points enclosed   : {inside} / {len(points)}",
        f"  bounding box      : x +- {np.sqrt(ell.covariance[0,0]):.5g},"
        f"  y +- {np.sqrt(ell.covariance[1,1]):.6g}",
        f"  correlation       : {ell.covariance[0,1] / np.sqrt(ell.covariance[0,0]*ell.covariance[1,1]):+.4f}",
    ]
    if level is not None:
        cov = ell.covariance / DELTA_CHI2_2DOF[level]
        lines += [
            f"  read as a {level}-sigma (2 d.o.f.) contour, the implied 1-sigma errors are",
            f"    sigma_x = {np.sqrt(cov[0,0]):.5g},  sigma_y = {np.sqrt(cov[1,1]):.6g}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def infer_level(path: str) -> int | None:
    import re

    m = re.search(r"(\d)\s*sigma", str(path), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datafile", nargs="+")
    ap.add_argument("--out", default=None, help="basename for the .pdf, .dat and .json outputs")
    ap.add_argument("--xlabel", default="x")
    ap.add_argument("--ylabel", default="y")
    ap.add_argument("--level", type=int, default=None, choices=[1, 2, 3],
                    help="n-sigma (2 d.o.f.) level; by default inferred from each filename")
    ap.add_argument("--trim", type=float, default=0.0,
                    help="fraction of the most extreme points to drop (e.g. 0.01)")
    ap.add_argument("--swap", action="store_true", help="swap the two columns")
    ap.add_argument("--method", default="mvee", choices=["mvee", "hull", "centre"],
                    help="which fit to export (default: the enclosing ellipse)")
    ap.add_argument("--reference", default=None, metavar="cx,cy,sx,sy",
                    help="published centre and 1-sigma errors, to check the axis calibration")
    args = ap.parse_args()

    results = {}
    for path in args.datafile:
        data = np.loadtxt(path)
        if args.swap:
            data = data[:, ::-1]
        level = args.level or infer_level(path)

        fits = {
            "outer envelope (minimum-volume enclosing ellipse)": outer_ellipse(data, "mvee", args.trim),
            "outer envelope (convex hull + conic fit)": outer_ellipse(data, "hull", args.trim),
            "band centre (conic fit to all points)": outer_ellipse(data, "centre"),
        }
        print(f"=== {path}  ({len(data)} points, level {level}) ===\n")
        for name, ell in fits.items():
            print(describe(name, ell, data, level))
            print()
        results[level] = dict(path=path, data=data, fits=fits,
                              chosen=fits[{"mvee": "outer envelope (minimum-volume enclosing ellipse)",
                                           "hull": "outer envelope (convex hull + conic fit)",
                                           "centre": "band centre (conic fit to all points)"}[args.method]])

    levels = sorted(k for k in results if k is not None)

    # ---- exported ellipse parameters -------------------------------------
    print("=" * 78)
    print("ELLIPSE PARAMETERS  (contour n-sigma, 2 d.o.f.;  method: %s)" % args.method)
    print("=" * 78)
    rows = []
    for lv in levels:
        e = results[lv]["chosen"]
        cov = e.covariance
        rows.append([lv, e.center[0], e.center[1], e.axes[0], e.axes[1], e.angle_deg,
                     cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])])
    hdr = ["n", "centre x", "centre y", "semi-major", "semi-minor", "angle [deg]", "corr"]
    widths = [max(len(h), 12) for h in hdr]
    print("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    for r in rows:
        print("  ".join(
            (f"{v:d}" if i == 0 else f"{v:.6g}").rjust(w) for i, (v, w) in enumerate(zip(r, widths))))

    if levels:
        print()
        print("Internal consistency: for 2 d.o.f. the linear size scales as sqrt(Delta chi^2)")
        ref = results[levels[0]]["chosen"]
        for lv in levels:
            e = results[lv]["chosen"]
            got = e.axes[0] / ref.axes[0]
            exp = np.sqrt(DELTA_CHI2_2DOF[lv] / DELTA_CHI2_2DOF[levels[0]])
            print(f"  {lv} sigma: size ratio {got:.3f}, expected {exp:.3f}"
                  f"   ({100*(got/exp-1):+.1f}%)")

        print()
        print("Implied 1-sigma errors from each level (they should agree):")
        for lv in levels:
            cov = results[lv]["chosen"].covariance / DELTA_CHI2_2DOF[lv]
            print(f"  from {lv} sigma: sigma_x = {np.sqrt(cov[0,0]):.5g}, "
                  f"sigma_y = {np.sqrt(cov[1,1]):.6g}, "
                  f"rho = {cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):+.4f}")
    print()

    if args.reference is not None and levels:
        cx, cy, sx, sy = (float(v) for v in args.reference.split(","))
        lv = levels[-1]
        e = results[lv]["chosen"]
        cov = e.covariance / DELTA_CHI2_2DOF[lv]
        fx, fy = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
        print("Comparison with the quoted published values")
        print(f"  centre x : fitted {e.center[0]:.5g}   published {cx:.5g}"
              f"   offset {(e.center[0]-cx)/sx:+.1f} published sigma")
        print(f"  centre y : fitted {e.center[1]:.6g}   published {cy:.6g}"
              f"   offset {(e.center[1]-cy)/sy:+.1f} published sigma")
        print(f"  sigma_x  : fitted {fx:.5g}   published {sx:.5g}   ratio {fx/sx:.2f}")
        print(f"  sigma_y  : fitted {fy:.6g}   published {sy:.6g}   ratio {fy/sy:.2f}")
        print(f"  area ratio (sigma_x sigma_y) : {(fx*fy)/(sx*sy):.2f}")
        if abs(e.center[0]-cx) > 2*sx or abs(e.center[1]-cy) > 2*sy:
            print("  -> the centre does not match: check the digitizer's axis calibration,")
            print("     or whether this contour belongs to a different data set.")
        print()

    # ---- outputs ----------------------------------------------------------
    if args.out:
        import json

        payload = {
            "source": {lv: results[lv]["path"] for lv in levels},
            "method": args.method,
            "convention": "n-sigma contours for 2 d.o.f. (Delta chi^2 = 2.30, 6.18, 11.83)",
            "xlabel": args.xlabel,
            "ylabel": args.ylabel,
            "ellipses": {},
        }
        for lv in levels:
            e = results[lv]["chosen"]
            payload["ellipses"][str(lv)] = {
                "center": e.center.tolist(),
                "semi_axes": e.axes.tolist(),
                "angle_deg": e.angle_deg,
                "contour_covariance": e.covariance.tolist(),
                "implied_1sigma_covariance": (e.covariance / DELTA_CHI2_2DOF[lv]).tolist(),
            }
            np.savetxt(f"{args.out}_{lv}sigma.dat", e.polyline(),
                       header=f"{args.method} ellipse for the {lv}-sigma contour\n"
                              f"{args.xlabel}  {args.ylabel}")
        with open(f"{args.out}.json", "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}.json and {args.out}_{{1,2,3}}sigma.dat")

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from reactor import plotting as pl
        pl.use_style()
        palette = [pl.BLUE, pl.ORANGE, pl.AQUA]
        ink = pl.INK
    except Exception:
        import matplotlib.pyplot as plt
        palette = ["tab:blue", "tab:orange", "tab:green"]
        ink = "k"

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for lv, color in zip(levels, palette):
        d = results[lv]["data"]
        ax.plot(d[:, 0], d[:, 1], ".", ms=2.0, color=color, alpha=0.45)
        xy = results[lv]["chosen"].polyline()
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.8, label=fr"${lv}\sigma$ (fitted)")
    if levels:
        ax.plot(*results[levels[0]]["chosen"].center, "+", color=ink, ms=10)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title("Outer ellipses fitted to the digitized contours")
    ax.legend(fontsize=9)
    if args.out:
        fig.savefig(f"{args.out}.pdf")
        print(f"wrote {args.out}.pdf")
    plt.show()


if __name__ == "__main__":
    main()
