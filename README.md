# Exoplanet Coronagraphy & Analysis Tools (ExoCAT) #
ExoCAT is a collection of codes and tools developed for the analysis of **JWST/MIRI** data.

## Contents
### MIRI coronagraphy
Tools developed for MIRI coronagraphic data analysis as part of the ExoMIRI GTO programs (GTO 1277 and GTO 1194; Boccaletti et al. 2024, Mâlin et al. 2024, Mâlin et al. 2025), and also applied to other MIRI coronagraphy programs (e.g., GO 3662; Lagrange et al. 2025).
Example notebooks for stellar PSF subtraction and planet photometry measurement.

### MIRI MRS
Notebooks with convenience functions for reading and handling MIRI MRS data cubes, as well as tools for spectral cross-correlation analysis (i.e. molecular mapping).
These tools have been used to analyze data from ERS 1386 (Mâlin et al. 2026), GO 3647 and GO 4829 (on-going analysis). More notebooks can be available upon request.

## Dependencies
ExoCAT relies on the following packages:
- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [astropy](https://www.astropy.org/)
- [matplotlib](https://matplotlib.org/)
- [photutils](https://photutils.readthedocs.io/)
- [webbpsf](https://webbpsf.readthedocs.io/)
- [jwst pipeline](https://jwst-pipeline.readthedocs.io/)
