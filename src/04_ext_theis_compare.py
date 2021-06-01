"""Comparison of Theis and extended Theis."""
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from anaflow import theis, ext_theis_2d

plt.style.use('default')
# mpl.rc("text", usetex=True)
mpl.rc('lines', linewidth=2.5)
plt.close("all")
fig = plt.figure(figsize=[5, 3])
ax = fig.add_subplot(1, 1, 1)

time_ticks = []
time_labels = [r"$t=10$s", r"$t=10$min", r"$t=10$h"]
time = [10, 600, 36000]        # 10s, 10min, 10h
rad = np.geomspace(0.05, 4.2)  # radial distance from pumping well in [0, 4]
S = 1e-4                       # storage
TG = 1e-4                      # the geometric mean of the transmissivity
len_scale = 10.0               # upper bound for the length scale
var = 0.5                      # variance of the log-conductivity
rate = -1e-4                   # pumping rate
TH = TG * np.exp(-var / 2)     # the harmonic mean of the conductivity

head_KG = theis(time, rad, S, TG, rate)
head_KH = theis(time, rad, S, TH, rate)
head_ef = ext_theis_2d(
    time=time,
    rad=rad,
    storage=S,
    trans_gmean=TG,
    len_scale=len_scale,
    var=var,
    rate=rate,
)
for i, step in enumerate(time):
    label_TG = "Theis($T_G$)" if i == 0 else None
    label_TH = "Theis($T_H$)" if i == 0 else None
    label_ef = "Extended Theis" if i == 0 else None
    color = "C0"  # + str(i + 1)
    ax.plot(
        rad, head_KG[i], label=label_TG, color=color, linestyle="--", alpha=0.7
    )
    ax.plot(
        rad, head_KH[i], label=label_TH, color=color, linestyle=":", alpha=0.7
    )
    ax.plot(rad, head_ef[i], label=label_ef, color=color)
    text_v = head_ef[i][-1]
    ax.annotate(
        time_labels[i],
        xy=(rad[-1], text_v),
        xytext=(rad[-1] + 0.1, text_v),
        verticalalignment="center",
    )
ax.set_xlim([-0.2, 4.2])
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_ylim([-1.7, 0.1])
ax.set_yticks([-1.5, -1, -0.5, 0])
ax.set_xlabel("$r$ in $[m]$")
ax.set_ylabel("$h(r)$ in $[m]$")
ax.grid(linestyle=':')
ax.legend()
ylim = ax.get_ylim()
fig.tight_layout()
fig.savefig(os.path.join("..", "results", "ext_theis_compare.pdf"), dpi=300)
fig.show()
