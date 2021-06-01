[![GS-Frame](https://img.shields.io/badge/github-GeoStat_Framework-468a88?logo=github&style=flat)](https://github.com/GeoStat-Framework)
[![Gitter](https://badges.gitter.im/GeoStat-Examples/community.svg)](https://gitter.im/GeoStat-Examples/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4891874.svg)](https://doi.org/10.5281/zenodo.4891874)

# The extended Theis solution in 2D


## Description

The extended Theis solution presented by *Zech et al. (2016)*
reproduces the ensemble mean drawdown
of pumping tests in heterogenous media with a log-normal transmissivity
distribution following a Gaussian variogram.

In this workflow, we demonstrate that the extended Theis solution reproduces
the ensemble mean drawdown of pumping tests on synthetic aquifers for multiple
parameter constellations.
These synthetic aquifers are created with
[GSTools](https://github.com/GeoStat-Framework/GSTools)
and the pumping tests are simulated by
[ogs5py](https://github.com/GeoStat-Framework/ogs5py).

The extended Theis solution was presented in:

> Zech, A., Müller, S., Mai, J., Heße, F., & Attinger, S., 2016.
> Extending theis’ solution: Using transient pumping tests to estimate parameters of aquifer heterogeneity.
> Water Resources Research 52, 6156–617. https://dx.doi.org/10.1002/2015WR018509


## Structure

The workflow is organized by the following structure:
- `src/` - here you should place your python scripts
  - `00_run_sim_mpi.sh` - bash file running `01_run_sim.py` in parallel
  - `01_run_sim.py` - run all ensemble simulations for pumping tests
  - `02_compare_mean.py` - generate comparision plots for the ensemble means
  - `03_trans_plot.py` - plot a realization of a gaussian transmissivity field
  - `04_ext_theis_compare.py` - plotting the effective Theis solution against classical Theis
- `results/` - all produced results


## Python environment

Main Python dependencies are stored in `requirements.txt`:

```
gstools==1.3.0
anaflow==1.0.1
ogs5py==1.1.1
matplotlib
```

You can install them with `pip` (potentially in a virtual environment):

```bash
pip install -r requirements.txt
```


## Contact

You can contact us via <info@geostat-framework.org>.


## License

MIT © 2021
