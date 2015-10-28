# syncological

Is a detailed look at the synchronization and neural code fidelity of gamma 
oscillations.

A 2015 SFN poster outlining the motivation, methods, and results of this work is available [here](https://github.com/voytekresearch/syncological/blob/master/analysis/SFN_POSTER_2015.pdf)

# Install

Simulation and initial analysis is in Python 2.7. Plotting and some light analysis is in R. Code has been tested *Nix systems, in OSX (10.11) and Linux (Ubuntu 4.10). It won't work on Windows.

## Python

1. Download or clone this repo to somewhere on your [python path](https://support.enthought.com/hc/en-us/articles/204469160-How-do-I-set-PYTHONPATH-and-other-environment-variables-for-Canopy-)
2. Run all experiments (via the command line) by callling `make sfn` from the top-level `syncological` directory.

Note: the experiments are run in parallel, on up to 12 cores (though this can expanded in the Makefile). If you have 12 cores available, as I do, these simulation will take several hours to complete.

## R

Open the `analysis/sfn_figures.Rmd` in Rstudio (see below) and run it.

## Dependencies.

The R and Python code has substantial dependencies. It you already have basic scientific installs of each, these should be easy to fulfill. That said, I've not created a proper install file so you'll have to do it by hand.   

The Python dependencies are:

- [Brian2](http://brian2.readthedocs.org/en/2.0b4/user/index.html)
- [numpy](http://www.numpy.org)
- [scipy](http://www.scipy.org)
- [fakespikes](https://github.com/voytekresearch/fakespikes)
- [PySpike](https://github.com/mariomulansky/PySpike)

For R based plotting I *highly* recommend you download and install [Rstudio](https://www.rstudio.com/products/RStudio/) from which is it very easy to install the following

- ggplot2
- grid
- gridExtra
- plyr
- dplyr
- reshape
- png
- psd
- tidyr
- doParallel
- bspec

The final 'system' dependency is GNU Parallel, which can be install on Linux via your package manager or on OSX by `brew install parallel`.
