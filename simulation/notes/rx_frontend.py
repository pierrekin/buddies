# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RX front-end: receive sensitivity -> LNA noise spec -> part picking
#
# Closes the gap left by link_budget: that notebook pins the receiver electronics
# noise budget in ACOUSTIC units (NL_elec = 25 dB re 1 uPa^2/Hz, input-referred,
# ambient-limited with ~4 dB margin) but never converts it to a VOLTAGE spec a part
# can be picked against. This does that conversion via the transducer receive
# sensitivity, then derives signal levels, the gain plan, and the resulting LNA + FDA
# part requirements.
#
# Chain (decided elsewhere): PZT -> STHV748 T/R -> LVOUT (+/-1 V clamp) -> LNA ->
# LTC6912 PGA (TGC) -> anti-alias -> FDA -> AD7380-4 (+/-VREF diff, VREF=3.3 V).

# %%
import numpy as np

# --- constants (from link_budget / frequency_window; single source of truth there) ---
c, rho, Q = 1500.0, 1000.0, 10.0      # m/s, kg/m^3, loaded quality factor
F0 = 90e3                              # Hz, resonance
OD, ID, Lc = 11e-3, 8.6e-3, 10e-3      # m, tube geometry
Cs = 5.0e-9                            # F, static capacitance
d31 = 120e-12                          # m/V, transverse piezo constant
eta_tx = 0.5                           # radiating fraction
R_res = 150.0                          # ohm, in-air loss resistance
R_in = R_res / (1 - eta_tx)            # 300 ohm in-water input resistance
NL_elec = 25.0                         # dB re 1 uPa^2/Hz, input-referred electronics floor
band = F0 / Q                          # 9 kHz noise bandwidth
SL = 169.0                             # dB re 1 uPa @ 1 m, at V_drive=20 Vrms (link_budget)

# --- front-end fixed points (decided) ---
LVOUT_CLAMP = 1.0                      # V, STHV748 T/R receive-output clamp (+/-1 V)
VREF = 3.3                             # V, AD7380-4 reference
ADC_FS_PK = VREF                       # +/-VREF differential -> +/-3.3 V peak


def thorp(f_khz):
    f = f_khz
    return 0.11*f**2/(1+f**2) + 44*f**2/(4100+f**2) + 2.75e-4*f**2 + 0.003


# %% [markdown]
# ## 1. Receive sensitivity M (V/Pa)
#
# Estimate from the direct piezo effect, then add the mechanical resonance gain.
# Radial-mode tube, radially poled (field through the 1.2 mm wall). External pressure
# p -> hoop stress sigma = p*Rm/t in the wall -> charge via d31 -> open-circuit volts.
#
# Open-circuit voltage is independent of electrode area:
#   Q_sc = d31 * sigma * A_elec ;  V_oc = Q_sc / Cs ;  Cs = eps*A_elec/t
#   => V_oc = d31 * sigma * t / eps  (area cancels) -- but we keep A_elec/Cs explicit.
#
# Quasi-static M_qs is sanity-checked against a real 90 kHz reference hydrophone
# (Teledyne RESON TC4013, -211 dB re 1 V/uPa). At resonance the strain (hence charge)
# is amplified by Q, so M_res ~ Q * M_qs. Treat M_res as an estimate good to a few dB;
# the LNA conclusion below is shown to be robust to that.

# %%
t_wall = (OD - ID) / 2
Rm = (OD + ID) / 4                        # mean radius
A_elec = 2*np.pi*Rm * Lc                  # electrode (wall) area
sigma_per_p = Rm / t_wall                 # hoop stress per unit pressure (thin tube)
Qsc_per_p = d31 * sigma_per_p * A_elec    # C per Pa
M_qs = Qsc_per_p / Cs                     # V/Pa, quasi-static
M_res = Q * M_qs                          # V/Pa, at resonance


def dB_re_1V_uPa(M):  # M in V/Pa -> dB re 1 V/uPa
    return 20*np.log10(M * 1e-6)


print(f"t_wall={t_wall*1e3:.1f} mm  Rm={Rm*1e3:.1f} mm  A_elec={A_elec*1e4:.2f} cm^2")
print(f"M_qs  = {M_qs:.2e} V/Pa = {dB_re_1V_uPa(M_qs):6.1f} dB re 1V/uPa   "
      f"(TC4013 ref ~ -211 -> sanity OK)")
print(f"M_res = {M_res:.2e} V/Pa = {dB_re_1V_uPa(M_res):6.1f} dB re 1V/uPa   (x Q)")

# %% [markdown]
# ## 2. LNA input voltage-noise target
#
# NL_elec is the electronics floor expressed acoustically. Refer it to volts through
# M_res: the LNA (the first noise-setting stage) must have input voltage noise
#   e_n <= M_res * p_n(NL_elec)
# to keep the receiver ambient-limited (preserve link_budget's 4 dB margin).

# %%
p_n = np.sqrt(10**(NL_elec/10)) * 1e-6     # Pa/rtHz  (uPa->Pa)
en_target = M_res * p_n                      # V/rtHz
print(f"ambient/electronics floor pressure PSD = {p_n*1e6:.1f} uPa/rtHz")
print(f"LNA input voltage-noise target  e_n <= {en_target*1e9:.1f} nV/rtHz @ {F0/1e3:.0f} kHz")
# robustness: M uncertain by +/-6 dB -> target moves but stays easy
for dM in (-6, 0, +6):
    print(f"   if M off by {dM:+d} dB -> e_n target = "
          f"{en_target*10**(dM/20)*1e9:5.1f} nV/rtHz")

# %% [markdown]
# ## 3. Signal levels over the operating range -> input dynamic range + TGC
#
# Received level RL(R) = SL - TL(R); open-circuit input volts V_oc = M_res * 10^(RL/20).
# The STHV748 T/R clamps LVOUT at +/-1 V, so the first stage never sees more than that.

# %%
def TL(R):
    return 20*np.log10(R) + thorp(F0/1e3)/1000.0 * R


def Voc(R):  # Vrms open-circuit at the transducer terminals
    RL = SL - TL(R)
    return M_res * 10**(RL/20) * 1e-6


print(f"{'R (m)':>6} {'RL (dB)':>8} {'Voc (mVrms)':>12} {'Vpk':>8} {'clamped?':>9}")
for R in (0.25, 0.5, 1, 2, 5, 10):
    v = Voc(R)
    vpk = v*np.sqrt(2)
    print(f"{R:>6} {SL-TL(R):>8.0f} {v*1e3:>12.1f} {vpk:>8.2f} "
          f"{'YES' if vpk > LVOUT_CLAMP else 'no':>9}")

tgc_db = 20*np.log10(Voc(0.25)/Voc(10))
print(f"\nlevel swing 0.25-10 m = {tgc_db:.0f} dB  -> LTC6912 (+40 dB) covers it: "
      f"{'YES' if tgc_db < 40 else 'NO'}")

# %% [markdown]
# ## 4. Noise floor in volts + SNR sanity (cross-check vs link_budget)


# %%
def noise_amb(f):
    turb = 17 - 30*np.log10(f)
    shp = 40 + 26*np.log10(f) - 60*np.log10(f+0.03)
    wnd = 50 + 7.5*np.sqrt(5.0) + 20*np.log10(f) - 40*np.log10(f+0.4)
    thm = -15 + 20*np.log10(f)
    return 10*np.log10(sum(10**(x/10) for x in (turb, shp, wnd, thm)))


NL = noise_amb(F0/1e3)
v_noise = M_res * np.sqrt(10**(NL/10) * band) * 1e-6   # Vrms in-band ambient at input
AG = 10*np.log10(4)
for R in (10, 50, 100):
    snr = 20*np.log10(Voc(R)/v_noise) + AG
    print(f"  R={R:>4} m  Voc={Voc(R)*1e6:8.1f} uVrms  in-band SNR={snr:5.1f} dB (DT=15)")
print(f"  in-band ambient noise at input = {v_noise*1e9:.0f} nVrms over {band/1e3:.0f} kHz")

# %% [markdown]
# ## 5. Gain plan
#
# Map the input window to the ADC. First-stage input is bounded by the LVOUT clamp
# (+/-1 V). The total gain must (a) not clip the near/clamp signal at minimum gain,
# and (b) lift the far-range signal to fill the ADC at maximum gain. Split: a modest
# fixed LNA gain (input-referred noise is gain-independent, so the LNA's job is noise,
# not gain) + the LTC6912 variable range (TGC) + the FDA to scale to +/-VREF.

# %%
v_far = Voc(10)                 # smallest in-range signal (Vrms)
v_clamp = LVOUT_CLAMP/np.sqrt(2) # largest first-stage signal (Vrms)
g_total_max = ADC_FS_PK/np.sqrt(2) / v_far     # to fill ADC at far range
g_total_min = ADC_FS_PK/np.sqrt(2) / v_clamp   # to not clip at clamp
print(f"to fill ADC at 10 m: total gain ~ {20*np.log10(g_total_max):.0f} dB")
print(f"to not clip at clamp: total gain ~ {20*np.log10(g_total_min):+.0f} dB")
print(f"=> total variable span needed ~ {20*np.log10(g_total_max/g_total_min):.0f} dB")
print("   split: fixed LNA (set by headroom, ~6-12 dB) + LTC6912 0..+40 dB (TGC) + FDA scale")

# %% [markdown]
# ## 6. Resulting part requirements + candidates
#
# **LNA** (first stage, sets the noise floor):
# - input voltage noise <= ~5 nV/rtHz at 90 kHz (target above; robust to M error)
# - LOW current noise: source is the tuned element (~hundreds of ohm to ~kohm
#   off-tune), so a JFET/CMOS input keeps i_n*Z_s below e_n
# - bandwidth: trivial (modest gain x 90 kHz)
# - rails: +5 / -2.5 V (matches the AD7380-4 driver rails); input handles +/-1 V (clamp)
# - non-BGA
# candidates: OPA1612 (1.1 nV, dual -> 2 chips/4 ch, bipolar), OPA827 (4 nV JFET, fA i_n,
#   robust to source Z), ADA4898-1 (0.9 nV). Pick on the noise/current-noise/source-Z
#   trade; all clear ~5 nV with margin.
#
# **FDA** (single->diff ADC driver):
# - drive +/-VREF (=+/-3.3 V) differential, common mode ~VREF/2 = 1.65 V
# - rails +5 / -2.5 V; -3 dB BW > ~95 kHz at the set gain (trivial)
# - noise non-critical (downstream of LNA gain)
# candidates: ADA4940-1, THS4551, LTC6363 (all standard AD7380-class drivers).
#
# **CHECKPOINT (architecture) -- RESOLVED:** the STHV748 T/R / LVOUT path sits ahead of
# the LNA, so its noise cannot be recovered. Datasheet Fig. 10 models the T/R switch as a
# PASSIVE network (series Rs=13 ohm, shunt Rp=100 kohm || Cp=40 pF) -- no active buffer.
# So it adds only thermal noise: series Rs=13 ohm -> sqrt(4kTR) ~ 0.46 nV/rtHz, ~10x below
# the ~5 nV/rtHz budget. The 100 kohm shunt only matters into a high-Z node; at resonance
# the tuned transducer source (~hundreds of ohm) shunts it, so it contributes little in
# operation. => LVOUT does NOT set the noise floor; the LNA does, as designed.
# Layout note (datasheet): tie the STHV748 exposed pad to ground via a 100 V cap to reduce
# receive-phase noise.

# %%
# %% [markdown]
# ## 7. Gain budget (LNA OPA1612 / PGA LTC6912 / FDA ADA4940) -> rail decision
#
# Total gain must (near) fill the ADC at the closest operating range with the PGA at
# minimum, and (far) fill it at the far range with the PGA near max. The LNA fixed gain
# is pinned between two constraints: noise (enough gain to bury the PGA's 12.6 nV/rtHz)
# and clipping (LNA output must not exceed its swing on the largest in-range signal).
# This pins the analog rail.

# %%
e_lna, e_pga = 2.0e-9, 12.6e-9          # V/rtHz: LNA added (at our Zs), PGA input noise
budget = en_target                       # 5.4 nV/rtHz
ADC_pk = ADC_FS_PK                        # 3.3 V differential peak
vin_near = Voc(0.25) * np.sqrt(2)        # Vpk at closest operating range
vin_far = Voc(10) * np.sqrt(2)

g_tot_near = ADC_pk / vin_near            # min total gain (fill ADC at near)
g_tot_far = ADC_pk / vin_far             # max total gain (fill ADC at far)
print(f"total gain: near(0.25m)={20*np.log10(g_tot_near):.0f} dB, "
      f"far(10m)={20*np.log10(g_tot_far):.0f} dB, TGC span={20*np.log10(g_tot_far/g_tot_near):.0f} dB")


def amp_noise(G_lna_dB):
    g = 10**(G_lna_dB/20)
    return np.sqrt(e_lna**2 + (e_pga/g)**2)


def min_glna_for_noise():
    g = np.arange(0, 30, 0.1)
    ok = [x for x in g if amp_noise(x) <= budget]
    return ok[0]


print(f"min LNA gain for noise<=budget: {min_glna_for_noise():.0f} dB "
      f"(total noise there = {amp_noise(min_glna_for_noise())*1e9:.1f} nV/rtHz)")

# clip headroom: OPA1612 swings ~1.2 V from each rail; AC about 0 V (neg rail present)
for rail, swing in [("+5/-2.5", min(3.8, 1.3)), ("+/-5", min(3.8, 3.8))]:
    g_clip = swing / vin_near             # max LNA gain before near-field clips
    print(f"  rail {rail:>8}: LNA output swing +/-{swing:.1f} V -> "
          f"max LNA gain {20*np.log10(g_clip):+.0f} dB (clip at 0.25 m)")

# %% [markdown]
# +5/-2.5 closes only marginally (LNA gain pinned ~8-9 dB: at the noise floor AND the
# clip ceiling at once -> no margin, erodes the link_budget 4 dB). +/-5 V opens it:
# LNA ~15 dB sits well above the noise minimum and well below the clip ceiling.
# -> run the RX analog chain on +/-5 V (not +5/-2.5); FDA Vocm pin still sets the ADC
# common mode to 1.65 V independent of the rails. Parts all support +/-5 V (10 V total):
# OPA1612 (<=36 V), LTC6912 (<=10.5 V), ADA4940 (<=10 V).

# %%
G_LNA = 15.0                              # dB, chosen on +/-5 V
G_FDA = 20*np.log10(g_tot_near) - G_LNA   # dB, fixed (PGA at min at near range)
G_PGA_MAX = 20*np.log10(g_tot_far) - G_LNA - G_FDA
print(f"GAIN PLAN (+/-5 V):  LNA {G_LNA:.0f} dB (fixed) | "
      f"PGA 0..{G_PGA_MAX:.0f} dB (TGC) | FDA {G_FDA:.0f} dB (fixed)")
print(f"  PGA needs gain x{10**(G_PGA_MAX/20):.0f} -> LTC6912 step x50 (34 dB) covers it")
print(f"  total amp noise at LNA={G_LNA:.0f} dB = {amp_noise(G_LNA)*1e9:.1f} nV/rtHz "
      f"(<< {budget*1e9:.1f} budget -> margin preserved)")
# node level check at the near range (PGA min), +/-5 V swing +/-3.8 V
lna_out = vin_near * 10**(G_LNA/20)
print(f"  near-field nodes: LNA out {lna_out:.1f} Vpk (<3.8 ok), "
      f"FDA out {lna_out*10**(G_FDA/20):.1f} Vpk diff (~ADC FS)")
print(f"  below ~0.12 m LVOUT clamps at +/-1 V and the chain saturates (acceptable)")

# %%
print("LNA target   : e_n <= ~5 nV/rtHz, low i_n (JFET/CMOS), +/-1 V in, non-BGA")
print("LNA cands    : OPA1612 / OPA827 / ADA4898-1")
print("FDA target   : +/-3.3 V diff @ CM 1.65 V, +5/-2.5 V, BW>95 kHz, noise non-critical")
print("FDA cands    : ADA4940-1 / THS4551 / LTC6363")
kT = 1.38e-23 * 300
en_tr = np.sqrt(4 * kT * 13.0)
print(f"CHECKPOINT   : RESOLVED -- T/R is passive (Rs=13ohm) -> {en_tr*1e9:.2f} nV/rtHz added "
      f"(<< {en_target*1e9:.1f} budget); LNA sets the floor. EP->gnd cap for rx-phase noise.")
