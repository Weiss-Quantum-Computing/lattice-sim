"""Echo-phase sensitivity to rotating-arm ellipticity, and the leg-to-leg
phase implied by a measured extinction profile.

S(theta) = d(Delta E)/d(eta), Delta E = E(|3,+m>) - E(|3,-m>) at each
partner's trap bottom (minimum + zero-point), x pair, 10 kHz lattice scale.
The echo cancels any Delta E common to both legs; a leg-to-leg change in
ellipticity eta leaves delta_phi = 2 pi * integral S(theta(t)) [eta_1 - eta_2] dt.

Usage: python scripts/echo_ellipticity_sensitivity.py ER_CSV
Writes data/echo_sensitivity.npz.
"""

import sys

import numpy as np

from lattice_sim.atom.ground_state import magic_field
from lattice_sim.fields.extinction import ExtinctionProfile, read_rampol_er
from lattice_sim.fields.lattice_pair import LatticePair
from lattice_sim.fields.potential import StatePotential

B = magic_field() * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
A = 4.8374e-6


def delta_e(theta, eta, m, handed=-1):
    er = np.inf if eta == 0 else 1 / eta**2
    model = StatePotential(LatticePair(extinction_ratio=er, handedness=handed), B)
    out = []
    for mm in (m, -m):
        sign = 1 if mm > 0 else -1
        guess = sign * A / 2 * np.sin(theta / 2) ** 2
        x0, f = model.minimum((3, mm), theta, guess=guess)
        out.append(model.potential((3, mm), np.array([x0]), theta)[0] + f / 2)
    return out[0] - out[1]


def sensitivity(m, thetas, eta=0.005):
    return np.array([(delta_e(t, eta, m) - delta_e(t, 0.0, m)) / eta for t in thetas])


def leg_trajectory(rows, leg, dt_s=1e-6):
    """theta(t) of one transport (up, hold, down) on a uniform grid, from the
    (t, rotation) of every row of that leg inside its up/hold/down segments."""
    pick = [r for r in rows if int(r["leg"]) == leg and
            r["segment"].split()[0] in ("up", "hold", "down")]
    t = np.array([r["t_ms"] for r in pick]) * 1e-3
    th = np.deg2rad(np.abs([r["rotation_deg"] for r in pick]))
    order = np.argsort(t)
    t, th = t[order], th[order]
    grid = np.arange(t[0], t[-1], dt_s)
    th_g = np.interp(grid, t, th)
    t_peak = t[np.argmax(th)]
    away = grid <= t_peak
    return grid - grid[0], th_g, away


def _sigma(rows, leg, direction, theta, er):
    """1-sigma of a dip ER point (mean of the asymmetric bars); lower bounds get 0."""
    for r in rows:
        if (r["method"] == "dip" and int(r["leg"]) == leg and abs(np.deg2rad(abs(r["rotation_deg"])) - theta) < 0.02
                and abs(r["er"] - er) < 1e-6 * er):
            if r["lower_bound"] == 1 or np.isnan(r["er_sigma_lo"]):
                return 0.0
            return 0.5 * (r["er_sigma_lo"] + r["er_sigma_hi"])
    return 0.0


def main(path):
    rows = read_rampol_er(path)
    thetas = np.deg2rad(np.arange(0, 181, 5.0))
    out = {"theta": thetas}
    for m in (3, 1):
        out[f"S_m{m}"] = sensitivity(m, thetas)
        print(f"|3,+/-{m}>: peak |S| {np.abs(out[f'S_m{m}']).max()/1e3:.0f} kHz per unit eta "
              f"at {np.rad2deg(thetas[np.argmax(np.abs(out[f'S_m{m}']))]):.0f} deg", flush=True)
    profiles = {(leg, d): ExtinctionProfile.from_rows(rows, leg, d)
                for leg in (1, 2) for d in ("away", "back")}
    t, th, away = leg_trajectory(rows, 1)
    dt = np.gradient(t)
    print(f"leg 1 transport: {t[-1]*1e3:.2f} ms, {np.sum(np.abs(th - np.pi/2) < np.deg2rad(20))*1e3*dt[0]:.2f} ms within 20 deg of pi/2")
    resolved = np.abs(th - np.pi / 2) > np.deg2rad(37.5)  # where both legs have bounded, not lower-bound, ER
    for m in (3, 1):
        s = np.interp(th, thetas, out[f"S_m{m}"])
        eta = {leg: np.where(away, profiles[(leg, "away")].eta(th), profiles[(leg, "back")].eta(th))
               for leg in (1, 2)}
        phase_leg = {leg: 2 * np.pi * np.sum(s * eta[leg] * dt) for leg in (1, 2)}
        dphi = phase_leg[1] - phase_leg[2]
        # weight of each angle: phase per unit eta accumulated there over one transport
        w = 2 * np.pi * np.sum(np.abs(s) * dt)
        near = np.abs(th - np.pi / 2) < np.deg2rad(20)
        w_near = 2 * np.pi * np.sum(np.abs(s[near]) * dt[near])
        d_res = 2 * np.pi * np.sum((s * (eta[1] - eta[2]) * dt)[resolved])
        print(f"|3,+/-{m}>: ellipticity phase per transport (upper bound, ER lower bounds near pi/2): "
              f"leg 1 {phase_leg[1]:+.1f} rad, leg 2 {phase_leg[2]:+.1f} rad")
        print(f"    leg 1 - leg 2 from angles where ER is resolved (|theta - pi/2| > 37.5 deg): {d_res:+.3f} rad")
        out[f"dphi_resolved_m{m}"] = d_res
        # uncertainty of the resolved difference from the dip error bars (Monte Carlo)
        rng = np.random.default_rng(1)
        draws = []
        for _ in range(400):
            eta_mc = {}
            for leg in (1, 2):
                vals = []
                for d in ("away", "back"):
                    p = profiles[(leg, d)]
                    sig = np.array([_sigma(rows, leg, d, t, e) for t, e in zip(p.theta, p.er)])
                    er_draw = np.clip(p.er + rng.normal(0, 1, p.er.size) * sig, 50, None)
                    vals.append(1 / np.sqrt(np.exp(np.interp(th, p.theta, np.log(er_draw)))))
                eta_mc[leg] = np.where(away, vals[0], vals[1])
            draws.append(2 * np.pi * np.sum((s * (eta_mc[1] - eta_mc[2]) * dt)[resolved]))
        print(f"    (Monte Carlo over dip error bars: {np.mean(draws):+.3f} +- {np.std(draws):.3f} rad)")
        e0 = np.array([delta_e(t, 0.0, m) for t in thetas])
        drag = 2 * np.pi * np.sum(np.interp(th, thetas, e0) * dt)
        print(f"    drag (ER-independent) differential phase per transport: {drag:+.1f} rad")
        out[f"drag_phase_m{m}"] = drag
        out[f"E0_m{m}"] = e0
        out[f"cum_drag_m{m}"] = 2 * np.pi * np.cumsum(np.interp(th, thetas, e0) * dt)
        out[f"cum_ell_m{m}"] = 2 * np.pi * np.cumsum(s * eta[1] * dt)
        print(f"    echo phase per unit common-mode eta change: {w:.0f} rad "
              f"({100*w_near/w:.0f} % from within 20 deg of pi/2); "
              f"0.1 rad needs |delta eta| < {0.1/w:.1e}")
        out[f"dphi_m{m}"] = dphi
        out[f"w_m{m}"] = w
    out["traj_t"], out["traj_theta"] = t, th
    np.savez("data/echo_sensitivity.npz", **out)


if __name__ == "__main__":
    main(sys.argv[1])
