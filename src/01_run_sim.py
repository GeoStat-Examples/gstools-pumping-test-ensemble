"""Generate an ensemble of drawdowns with ogs5py and GSTools."""
import os
import shutil
import numpy as np
from ogs5py import OGS, specialrange, generate_time, by_id
from ogs5py.reader import readpvd
import gstools as gs
from mpi4py import MPI


def angles_mean(time, rad, angles, path):
    """Generate mean along angles for single simulation."""
    # read output from ogs5py
    out = readpvd(task_root=path, task_id="model", pcs="GROUNDWATER_FLOW")
    rt_head = np.zeros(time.shape + rad.shape, dtype=float)
    # loop over time
    for select, step in enumerate(time):
        points = out["DATA"][select]["points"]
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        rad_ids = np.argmin(np.abs(np.subtract.outer(rad, radii)), axis=0)
        for i, head in enumerate(out["DATA"][select]["point_data"]["HEAD"]):
            rt_head[select, rad_ids[i]] += head
    # only one head value for rad=0
    rt_head[:, 1:] = rt_head[:, 1:] / angles
    np.savetxt(os.path.join(path, "rad_mean_head.txt"), rt_head)


# rank is the actual core-number, size is total number of cores
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# state if OGS5 output files should be kept
keep_output = False
# size of the ensembles
ens_size = 1000
# pumping rate (1L / s)
prate = -1e-4
# parameter lists to generate the para_set (single one)
S = [1e-4]
TG = [1e-4]  # mu = log(TG)
var = [1.0, 2.25]
len_scale = [10, 20]
para_set = np.array(
    [[s, t, v, ls] for t in TG for v in var for ls in len_scale for s in S]
)

RES = os.path.join("..", "results")
# ogs configuration
task_root = os.path.abspath(os.path.join(RES, "ext_theis_2D"))
pcs_type_flow = "GROUNDWATER_FLOW"
var_name_flow = "HEAD"

# generate a model for each core (prevent writing conflicts)
cstr = "core{:04}".format(rank)
model = OGS(task_root=os.path.join(task_root, cstr), task_id="model")

# spatio-temporal configuration
# define the time stepping: 2 h with 32 steps and increasing stepsize
time = specialrange(0, 7200, 32, typ="cub")
# radial discretization: 1000 m with 100 steps and increasing stepsize
rad = specialrange(0, 1000, 100, typ="cub")
# 64 angles for discretization
angles = 64

# generate mesh and gli
model.msh.generate("radial", dim=2, angles=angles, rad=rad)
model.gli.generate("radial", dim=2, angles=angles, rad_out=rad[-1])
# add the pumping well
model.gli.add_points(points=[0.0, 0.0, 0.0], names="pwell")

# --------------generate different ogs input settings------------------------ #

model.pcs.add_block(  # set the process type
    PCS_TYPE=pcs_type_flow, NUM_TYPE="NEW"
)
model.mpd.add(name="transmissivity")
model.mpd.add_block(  # edit recent mpd file
    MSH_TYPE=pcs_type_flow, MMP_TYPE="PERMEABILITY", DIS_TYPE="ELEMENT",
)
model.mmp.add_block(  # permeability, storage and porosity
    GEOMETRY_DIMENSION=2,
    PERMEABILITY_TENSOR=["ISOTROPIC", 1.0],
    PERMEABILITY_DISTRIBUTION=model.mpd.file_name,
)
model.bc.add_block(  # set boundary condition
    PCS_TYPE=pcs_type_flow,
    PRIMARY_VARIABLE=var_name_flow,
    GEO_TYPE=["POLYLINE", "boundary"],
    DIS_TYPE=["CONSTANT", 0.0],
)
model.ic.add_block(  # set the initial condition
    PCS_TYPE=pcs_type_flow,
    PRIMARY_VARIABLE=var_name_flow,
    GEO_TYPE="DOMAIN",
    DIS_TYPE=["CONSTANT", 0.0],
)
model.st.add_block(  # set pumping condition at the pumpingwell
    PCS_TYPE=pcs_type_flow,
    PRIMARY_VARIABLE=var_name_flow,
    GEO_TYPE=["POINT", "pwell"],
    DIS_TYPE=["CONSTANT_NEUMANN", prate],
)
model.num.add_block(  # set the parameters for the solver
    PCS_TYPE=pcs_type_flow, LINEAR_SOLVER=[2, 5, 1.0e-14, 1000, 1.0, 100, 4],
)
model.tim.add_block(  # set the TIMESTEPS
    PCS_TYPE=pcs_type_flow, **generate_time(time)
)
model.out.add_block(  # set the outputformat for the whole domain
    PCS_TYPE=pcs_type_flow,
    NOD_VALUES=var_name_flow,
    GEO_TYPE="DOMAIN",
    DAT_TYPE="PVD",
    TIM_TYPE=["STEPS", 1],
)

# --------------run OGS simulation------------------------------------------- #

print("write files on core {:02}".format(rank))
model.write_input()

# save meta info only on core 0
if rank == 0:
    np.savetxt(os.path.join(task_root, "time.txt"), time)
    np.savetxt(os.path.join(task_root, "rad.txt"), rad)
    np.savetxt(os.path.join(task_root, "angles.txt"), [angles])
# collect failed runs
FAIL = []
# use a rank dependently seeded pseudo-random number generator
seed = gs.random.MasterRNG(rank)
for para_no, para in enumerate(para_set):
    if rank == 0:
        print("PARA_SET {:04}".format(para_no))
    # set storativity
    model.mmp.update_block(STORAGE=[1, para[0]])
    model.mmp.write_file()
    # init cov model (truncated power law with gaussian modes)
    cov = gs.Gaussian(dim=2, var=para[2], len_scale=para[3])
    # init spatial random field class
    srf = gs.SRF(
        model=cov,
        mean=np.log(para[1]),
        normalizer=gs.normalizer.LogNormal,
        upscaling="coarse_graining",
    )
    # run the ensemble
    for i in range(ens_size):
        # parallel running the right jobs on each core
        if (para_no * ens_size + i) % size != rank:
            continue
        # generate new transmissivity field
        srf.mesh(model.msh, seed=seed(), point_volumes=model.msh.volumes_flat)
        # add the transmissivity to the ogs project
        model.mpd.update_block(DATA=by_id(srf.field))
        # write the new mpd file
        model.mpd.write_file()
        # set the new output-directory
        model.output_dir = os.path.join(
            task_root, "para{:04}".format(para_no), "seed{:04}".format(i)
        )
        success = model.run_model(print_log=False, save_log=keep_output)
        print("  run model {:04}".format(i), end=" ")
        print("  ...success") if success else print("  ...error!")
        if not success:
            FAIL.append(str(para_no) + "_" + str(i))
        # calculate angular means
        angles_mean(time, rad, angles, model.output_dir)
        # export the generated transmissivity field as vtk
        if keep_output:
            model.msh.export_mesh(
                os.path.join(model.output_dir, "field.vtu"),
                file_format="vtk",
                cell_data_by_id={"transmissivity": srf.field},
            )
        else:
            files = model.output_files(pcs="GROUNDWATER_FLOW", typ="PVD")
            files.append("model_GROUNDWATER_FLOW.pvd")
            for file in files:
                os.remove(os.path.join(model.output_dir, file))
    if rank == 0:
        np.savetxt(  # save current parameter set to file
            os.path.join(task_root, "para{:04}".format(para_no), "para.txt"),
            para,
            header="storage, trans_gmean, var, len_scale",
        )
# remove OGS5 settings
if not keep_output:
    shutil.rmtree(os.path.join(task_root, cstr))
# final success message
if FAIL:
    print("core {:02} FAILED:".format(rank), FAIL)
else:
    print("core {:02} SUCCESS".format(rank))
