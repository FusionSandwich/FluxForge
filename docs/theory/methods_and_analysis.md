# Metrology Methods and Analysis Theory

## Overview
FluxForge is heavily bound by computational and particle physics models for accurately converting raw voltage signals on HPGe detectors into fully unfolved neutron spectra. This document gives brief oversight to our key metrological equations.

## Peak Identification (ASTM E3376)
- **Continuum Subtraction:** Uses the standard Covell Model (1959) which implements a two-point linear channel regression strictly bounded by the statistical Full-Width-Half-Maximum (FWHM).
- **Two-Stream Smoothness:** Peak counts use pure integer signals to enforce strict Poisson distribution parity, while Savitzky-Golay quadratic filters (generally over 5-7 channels) resolve underlying peaks visually. 

## Spectrum Unfolding 
- **STAYSL / GLS:** Generalized Least Squares regression mapping reaction rates back into probable continuous flux environments using prior fluxes and specific cross-section responses (IRDFF-II).
- **Iterative GRAVEL & MLEM:** Deterministic unfolding avoiding initial prior contamination, executing until theoretical folded fluxes reliably mirror measured saturation states.

## Decay Physics
- Follows universally recognized Bateman equations.
- Considers standard cooling intervals, beam variance, detector geometries, and baseline half-life definitions inherited directly from ENDF repositories.
