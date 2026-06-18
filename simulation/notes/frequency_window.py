# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Frequency window: range vs angular resolution
#
# Seawater, half-duplex quad array, envelope-TDOA bearing.
# Transverse resolving baseline fixed at D = 85 mm (85x45 element
# rectangle, long axis normal to target). c = 1500 m/s, Q = 10.
#
# Two tradeoffs, opposite sign, coupled through one absorption term:
# - reach wants low f (absorption)
# - angular resolution wants high f (aperture in wavelengths)

# %%
import numpy as np
import matplotlib.pyplot as plt

c, D, Q = 1500.0, 0.085, 10.0  # m/s, m, -


def thorp(f):  # f in kHz -> absorption in dB/km
    return 0.11 * f**2 / (1 + f**2) + 44 * f**2 / (4100 + f**2) + 2.75e-4 * f**2 + 0.003


def tl(f, R):  # transmission loss, dB; R in m, f in kHz
    return 20 * np.log10(R) + thorp(f) / 1000.0 * R


# %% [markdown]
# ## Absorption (Thorp)
#
# MgSO4 relaxation dominates 10-300 kHz, knee ~60 kHz.
# Rises ~f^2 at the low end, bends toward linear.
# Anchors (dB/km): 10 kHz ~1, 30 ~8, 90 ~32, 200 ~51.

# %%
f = np.linspace(1, 300, 600)
plt.figure(figsize=(6, 4))
plt.loglog(f, thorp(f))
plt.xlabel("f (kHz)")
plt.ylabel("alpha (dB/km)")
plt.title("seawater absorption (Thorp)")
plt.grid(True, which="both", alpha=0.3)

# %% [markdown]
# ## Range vs f
#
# TL = 20 log10 R + alpha(f) R/1000. Reach = R where TL hits the budget.
# Shape: plateau below ~20 kHz (spreading-limited, absorption negligible
# over a km), knee, then ~1/f^2 collapse. Lower f = more reach.

# %%
def reach(f, budget_db):
    R = np.logspace(0, 4, 4000)  # 1 m .. 10 km
    out = np.empty_like(f)
    for i, a in enumerate(thorp(f) / 1000.0):
        k = np.searchsorted(20 * np.log10(R) + a * R, budget_db)
        out[i] = R[min(k, len(R) - 1)]
    return out


f = np.linspace(5, 300, 400)
plt.figure(figsize=(6, 4))
for B in (60, 80, 100):
    plt.loglog(f, reach(f, B), label=f"{B} dB budget")
plt.xlabel("f (kHz)")
plt.ylabel("max range (m)")
plt.title("reach vs f: plateau, knee, 1/f^2 collapse")
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# %% [markdown]
# ## Angular resolution vs f (D = 85 mm)
#
# err ~ sigma_tau c / D, with sigma_tau ~ Q / (f sqrt(SNR)).
# High-SNR limit: err ~ 1/f, equivalently resolution = D/lambda = D f / c.
# 85 mm spans 3 lambda only above ~50 kHz, so usable bearing needs
# f >~ 50 kHz. Left panel is calibrated; right is shape only.

# %%
f = np.linspace(5, 300, 400)
lam = c / (f * 1e3)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(f, D / lam)
ax[0].axhline(3, ls="--", color="k", alpha=0.5)
ax[0].set(xlabel="f (kHz)", ylabel="D/lambda", title="aperture in wavelengths (85 mm)")
ax[1].loglog(f, 50.0 / f)  # err ~ 1/f, normalized to f=50 kHz
ax[1].set(xlabel="f (kHz)", ylabel="err / err(50 kHz)", title="high SNR: err ~ 1/f")
for a in ax:
    a.grid(True, which="both", alpha=0.3)

# %% [markdown]
# ## Coupling: the resolution curve is a bowl
#
# sigma_tau carries sqrt(SNR), and absorption sets SNR ~ 10^(-TL/10).
# Short range: SNR flat, err ~ 1/f, monotone, minimum at high f.
# Long range: absorption starves high-f SNR, err turns back up, minimum
# slides down. min(R=2 m) > 300 kHz; min(R=1 km) ~ 22 kHz.
# Each curve normalized to its own minimum to expose the shape.

# %%
f = np.linspace(20, 300, 400)
plt.figure(figsize=(6, 4))
for R in (2, 50, 200, 1000):
    snr = 10 ** (-tl(f, R) / 10)
    err = Q / (f * 1e3 * np.sqrt(snr)) * c / D
    plt.semilogy(f, err / err.min(), label=f"R = {R} m")
plt.xlabel("f (kHz)")
plt.ylabel("err / min (shape)")
plt.title("bearing err vs f: bowl minimum slides down with range")
plt.legend()
plt.grid(True, which="both", alpha=0.3)

# %% [markdown]
# ## Window
#
# - lower bound: aperture in wavelengths, f >~ 40-50 kHz (85 mm = 3 lambda).
# - upper bound: absorption, slides down with design range.
# - 2 m: window wide, high f fine.
# - 1 km: window pinches to tens of kHz.
