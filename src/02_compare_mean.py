"""
Post processing the ensembles.

Gau in [Attinger 2003]: exp(-0.5 * (r/l)**2)
Gau in [GSTools]: exp(-pi/4 * (r/l)**2)
"""
import os
import glob
import numpy as np
from ogs5py.reader import readpvd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm, rc
import anaflow as ana
from PyPDF2 import PdfFileMerger


rc("text", usetex=True)
CWD = os.path.abspath(os.path.join("..", "results"))


def get_rad(points):
    """Calculate radius of given points."""
    return np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)


def single_mean(para_start=0, base="ext_theis2d"):
    """Generate mean along angles for single simulation."""
    para_sets = sorted(glob.glob(os.path.join(CWD, base, "para*")))
    time = np.loadtxt(os.path.join(CWD, base, "time.txt"))
    rad = np.loadtxt(os.path.join(CWD, base, "rad.txt"))
    angles = int(np.loadtxt(os.path.join(CWD, base, "angles.txt")))

    for para_no, para_set in enumerate(para_sets):
        if para_no < para_start:
            continue
        np.loadtxt(os.path.join(para_set, "para.txt"))
        print(para_no, "PARA_SET")
        ensemble = sorted(glob.glob(os.path.join(para_set, "seed*")))

        for cnt, single in enumerate(ensemble):
            print("  ", cnt, single)
            out = readpvd(
                task_root=single, task_id="model", pcs="GROUNDWATER_FLOW"
            )
            rt_head = np.zeros(time.shape + rad.shape, dtype=float)

            for select, step in enumerate(time):
                radii = get_rad(out["DATA"][select]["points"])
                rad_ids = np.argmin(
                    np.abs(np.subtract.outer(rad, radii)), axis=0
                )

                for i, head in enumerate(
                    out["DATA"][select]["point_data"]["HEAD"]
                ):
                    rt_head[select, rad_ids[i]] += head

            # this is wrong for rad=0 (single value only, but ignored later)
            rt_head /= angles
            # store time and rad as well
            rt_head[:, 0] = time
            rt_head[0, :] = rad
            np.savetxt(os.path.join(single, "rad_mean_head.txt"), rt_head)


def global_mean(para_start=0, base="ext_theis2d"):
    """Generate mean for all simulations in ensemble from single means."""
    para_sets = sorted(glob.glob(os.path.join(CWD, base, "para*")))
    time = np.loadtxt(os.path.join(CWD, base, "time.txt"))
    rad = np.loadtxt(os.path.join(CWD, base, "rad.txt"))

    for para_no, para_set in enumerate(para_sets):
        if para_no < para_start:
            continue
        np.loadtxt(os.path.join(para_set, "para.txt"))
        print(para_no, "PARA_SET")
        ensemble = sorted(glob.glob(os.path.join(para_set, "seed*")))
        rt_head = np.zeros(time.shape + rad.shape, dtype=float)

        for cnt, single in enumerate(ensemble):
            sgl_head = np.loadtxt(os.path.join(single, "rad_mean_head.txt"))
            print("  ", cnt, single)
            rt_head += sgl_head

        rt_head /= len(ensemble)
        rt_head[:, 0] = time
        rt_head[0, :] = rad
        np.savetxt(os.path.join(para_set, "rad_mean_head.txt"), rt_head)


def compare(para_start=0, base="ext_theis2d"):
    """Compare ensemble mean to effective drawdown solution."""
    path = os.path.join(CWD, base)
    para_sets = sorted(glob.glob(os.path.join(path, "para*")))
    time = np.loadtxt(os.path.join(path, "time.txt"))
    rad = np.loadtxt(os.path.join(path, "rad.txt"))
    time_range = time > 60
    rad_range = np.logical_and(rad > 0.2, rad < 40)

    time_select = time[time_range]
    rad_select = rad[rad_range]

    for para_no, para_set in enumerate(para_sets):
        if para_no < para_start:
            continue
        para = np.loadtxt(os.path.join(para_set, "para.txt"))
        rt_head = np.loadtxt(os.path.join(para_set, "rad_mean_head.txt"))
        print(para_no, "PARA_SET")
        rt_head = rt_head[time_range]
        rt_head = rt_head[:, rad_range]
        et_head = ana.ext_theis_2d(
            time_select,
            rad_select,
            para[3],
            para[0],
            para[1],
            para[2],
            -1e-4,
            parts=50,
            prop=np.sqrt(np.pi * 2),  # the right one?
        )
        plot_diff(
            time_select, rad_select, rt_head, et_head, para_no, path, para
        )
    # merge pdfs
    pdfs = sorted(glob.glob(os.path.join(path, "*_diff.pdf")))
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(os.path.join(path, "diff.pdf"))


def plot_diff(time, rad, rt_head, et_head, para_no, path, para):
    """Plot the comparisson between effective head and ensemble mean."""
    plt.close("all")
    fig = plt.figure(figsize=[9, 6])
    ax0 = plt.subplot2grid((2, 3), (0, 0), fig=fig, projection=Axes3D.name)
    ax1 = plt.subplot2grid((2, 3), (1, 0), fig=fig, projection=Axes3D.name)
    ax2 = plt.subplot2grid(
        (2, 3), (0, 1), fig=fig, rowspan=2, colspan=2, projection=Axes3D.name
    )

    time_m, rad_m = np.meshgrid(time, rad, indexing="ij")
    diff = np.abs(rt_head - et_head)

    z_max = np.max((np.max(rt_head), np.max(et_head), np.max(diff)))
    z_min = np.min((np.min(rt_head), np.min(et_head), np.min(diff)))
    z_diff = z_max - z_min
    z_max += z_diff * 0.1
    z_max -= z_diff * 0.1
    z_diff_r = z_diff * 0.6

    diff_mean = (np.max(diff) - np.min(diff)) / 2

    ax0.plot_surface(
        rad_m,
        time_m,
        rt_head,
        rstride=1,
        cstride=1,
        cmap=cm.RdBu,
        linewidth=0.3,
        antialiased=True,
        edgecolors="k",
        label="h sim",
    )
    ax1.plot_surface(
        rad_m,
        time_m,
        et_head,
        rstride=1,
        cstride=1,
        cmap=cm.RdBu,
        linewidth=0.3,
        antialiased=True,
        edgecolors="k",
        label="h CG",
    )
    ax2.plot_surface(
        rad_m,
        time_m,
        diff,
        rstride=1,
        cstride=1,
        cmap=cm.RdBu,
        linewidth=0.3,
        antialiased=True,
        edgecolors="k",
    )
    ax2.plot_surface(
        rad_m,
        time_m,
        np.zeros_like(diff),
        rstride=1,
        cstride=1,
        color="k",
        alpha=0.4,
        antialiased=True,
    )
    ax0.view_init(elev=15, azim=-150)
    ax1.view_init(elev=15, azim=-150)
    ax2.view_init(elev=15, azim=-150)
    fig.suptitle(
        r"$T_G={:.1e}".format(para[0])
        + r"$, $\sigma^2={}".format(para[1])
        + r"$, $\ell={}".format(para[2])
        + r"$, $S={:.1e}".format(para[3])
        + r"$ "
        + r"(Set {:04}".format(para_no)
        + r")"
    )
    xlab = r"$r$ in $\mathrm{[m]}$"
    ylab = r"$t$ in $\mathrm{[s]}$"
    zlab_diff = r"$\Delta (h_\mathrm{sim},h_\mathrm{CG})$"
    # ax0.set_title(r"$h_\mathrm{sim}$", pad=-10)
    # ax1.set_title(r"$h_\mathrm{CG}$", pad=-10)
    # ax2.set_title(zlab_diff, pad=-15)
    ax0.set_title("Ensemble mean", pad=-10)
    ax1.set_title("Effective head", pad=-10)
    ax2.set_title("Absolute difference", pad=-15)
    # change axes labels
    ax0.set_xlabel(xlab)
    ax0.set_ylabel(ylab)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab)
    ax2.set_zlabel(zlab_diff + r" in $\mathrm{[m]}$")
    ax0.set_zlim((z_min, z_max))
    ax1.set_zlim((z_min, z_max))
    ax2.set_zlim((diff_mean - z_diff_r, diff_mean + z_diff_r))
    fig.tight_layout()
    plt.savefig(os.path.join(path, "{:04}_diff.pdf".format(para_no)))
    # plt.close("all")


# single_mean()
# global_mean()
compare()
