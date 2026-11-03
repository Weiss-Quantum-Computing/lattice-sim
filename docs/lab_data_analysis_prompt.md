# Prompt: analysing new molasses / hopping / imaging data

Paste the block below into a new session (Claude Code in the lattice-sim repo, or a linked Claude session), then fill in
the last section.

```text
You are analysing new lab data from the Weiss-group Cs 3D optical lattice (843.222 nm, blue detuned) for the imaging /
cooling molasses work. Repository: Weiss-Quantum-Computing/lattice-sim, branch molasses-lab-analysis (not yet merged).
Read CLAUDE.md, then docs/findings.md sections dated 2026-10-29 and 2026-10-30, before touching the data. Follow the
repo rules: every number backed by a committed script, tests pass before every commit, findings in dated sections in
scientific style (concise, quantitative, file that backs each statement), figures with lattice_sim.viz.style.apply()
(panel letters, units, colour-blind-safe), commit messages with the key numbers. Report a failed model comparison as
failed; do not tune the model to fit.

Apparatus facts used so far
- Lattices: X, X2 (alternative x lattice, used for state detection), Y, Z; spacing 4.8374 um; full-depth barriers from
  the model 270 / 192 / 117 uK (x / y / z). Planes are z; analysis uses planes 1-7 of the 9x9x9 block.
- Molasses: six retro-reflected beams (3 pairs, lin-perp-lin, LC phase modulation), one beam known misaligned. Lock on
  the F=4 -> F'=4/5 crossover (-125.5 MHz from F'=5) plus the AOM: +100 MHz -> -25.5 MHz (current); +95 -> -30.5,
  +105 -> -20.5, +75 -> -50.5 (old). Gamma/2pi = 5.234 MHz.
- Power: Supertime "Molasses beams" request = fiber-total uW with the calibration file OM-2026-10-29.txt (cut at
  4.4 V = 20.91 uW, AOMrescale = 20.91). Fiber power is flat with AOM frequency (19.5-21.0 uW at 4.192 V).
  1 uW fiber total ~ 0.71 mW/cm^2 per beam (Li's 140 um waist). Old-file requests at rescale 24 map to 5 -> 2.05,
  2 -> 0.66, 0.6 -> 0.13 uW.
- Helmholtz zero corrected on 2026-10-30 (offsets -11.5, -34.2, +30.7 mG applied to x, y, z); earlier runs used a zero
  ~47 mG away.
- Sequence: supertime-analysis repo, hopping_2EOx_2imgsets_9 planes.dt4 with the hold block between the image sets
  (5 ms lattice ramps, hold depth / field / molasses cells). Each day's log.txt says what was varied per run.

Data and files
- Day folders: ~/Desktop/Data/Camera Control/<YYYYMMDD> (the CameraGui folder date can lag: 2026-10-30 data are in
  20261029). If the data are only on the flash drive (/Volumes/MAARTEN USB/Data/...): NEVER run shell commands on the
  drive (it stays locked and will not eject). List it with the device list tool, copy files with the stage tool, write
  them to the Desktop folder with the commit tool. Copy the analysis files (.hist, .atom, .stat, .histsub, .state,
  .atomhack, *-analysis.json, *-hopevents.txt, *-LogHist*.txt, *processed.txt, *-fitlog.txt, *-DisplaceHistLog.txt,
  log.txt, analysislog.txt), not the raw .txt images or SVGs.
- .hist: per-site fit heights, shots x 2 images x 9 x 9 x 9. .atom: header with the camera cutoffs ("thresholds ="),
  then Before / After occupancy per shot. .stat / NNNprocessed.txt: the camera's per-point ratios. analysislog.txt:
  the scan (range, points x repeats); shots run sweep after sweep, point = shot mod n_points. .state: "After 1"
  state call (1 / 2) and "Displace 1" floating-centre displacement (px) in the state image; calibrated state
  displacements +1.41 px (I) and -2.28 px (II), boundary 0.
- Tool: python scripts/camera_runs.py DATA_DIR RUN [RUN ...] [--scan START STOP NPTS] [--state] [--out f.json]
  re-thresholds each run at the histogram valley (per image and neighbour category), gives brightness, separation,
  expected miscalls, hops / losses per atom of image 1, nearest-neighbour hop-loss pairs by direction against chance,
  loss against plane / row / column, per-point tallies for scans, and the displacement statistics. It drops shots with
  > 50 "hops" (misregistration or the end of an aborted acquisition: empty image 2, then a new loading).

Established results to compare against (docs/findings.md has the sources)
- Brightness is linear in power x exposure: ~7200 counts per uW per 100 ms at -25.5 MHz in X; the 14 500-count
  readout threshold needs ~0.20 uW s per exposure. Hops vs brightness at 100 ms nearly detuning-independent over
  -20.5 to -30.5 MHz; brightness per uW across detuning is NOT Lorentzian (1.37 : 1 : 1.04) and unexplained.
- 5 s hold with molasses at -25.5 MHz (old field zero): survival >= 0.85 for 0.7-2.9 uW (X) and 1.4-2.4 uW (X2);
  total loss for 0.1-0.4 uW (X) and 0.1-1.0 uW (X2); hops rise above 2.6 / 2.4 uW. Hop onset gives kT/U = 0.086 at
  2 uW in X (T ~ 10 uK on the z barrier; model 13 uK), attempt rate ~170 s^-1. X2 hops favour +x (only with molasses
  on); X hops favour -z. Dark loss ~1.4 %/s in X, ~4-5 %/s in X2.
- PGC field at 0.5 uW, 5 s, X: flat-topped survival (+-40 mG), half maximum near +-60-85 mG; correcting all three
  axes raised survival 0.67 -> 0.82 (plateau 0.90). Loss gradients across the block reverse between low and high
  power: a ~20 % molasses intensity gradient (brighter toward plane 6-7, row 8).
- X2 imaging at 0.20 uW s (both sets in X2): displacement tail |d| > 1 px 6.6 % (100 ms x 2.0 uW), 2.3 %
  (115 x 1.75), 1.8 % (135 x 1.5); beyond the state-II boundary 2.0 / 0.65 / 0.37 %; hops 1.1-1.2 %, losses
  3.8-5.9 %; separation in X2 4.0-4.6 sigma (X: 5.4-6.0); tail grows toward plane 7.
- Model (src/lattice_sim/molasses, tables in data/molasses): temperature per scattering rate rises ~linearly with
  |detuning| (5.0, 5.7, 6.9, 7.7 uK per 10^6 s^-1 at -15, -20, -25.5, -30 MHz); it is ~30 % warm at -25.5 MHz and
  fails the low-power loss band and the field sensitivity (2-3x too insensitive).

Open questions this data may address
- Detuning: true line shape / lock offset (brightness vs AOM frequency at fixed low power, Lorentzian fit),
  temperature per photon vs detuning (hold scans in X2 at -20.5 / -15.5 MHz on a common scattering-rate axis), and
  where the low-power loss edge sits in scattering rate.
- X2 imaging: whether longer exposure / lower power (e.g. 150 ms x 1.35 uW) cuts the displacement tail further
  without hitting the loss band; why X2 readout separation is lower.

New data
- Date / folder: <...>
- Runs and what was varied (from log.txt): <...>
- Question to answer: <...>
Steps: copy the data as above, run scripts/camera_runs.py, check each run for bad shots and scan mapping, compare with
the results above, make a figure script and data CSV under data/measured/, add a dated findings section, commit on the
branch, and give me a short scientific summary with the recommendation and its uncertainty.
```
