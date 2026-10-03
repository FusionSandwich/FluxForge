import numpy as np
import pytest
from fluxforge.io.spe import GammaSpectrum
from fluxforge.io.flux_wire import FluxWireData
from fluxforge.analysis.flux_wire_analysis import GammaLine, analyze_raw_spectrum_targeted


def synthetic_data(sigma_factor=1.0, multiplet=False):
    x=np.arange(512,dtype=float)
    counts=40.0+2000.0*np.exp(-0.5*((x-200.0)/1.2)**2)
    if multiplet:
        counts+=1000.0*np.exp(-0.5*((x-205.0)/1.2)**2)
    spectrum=GammaSpectrum(counts=counts,counts_uncertainty=np.sqrt(counts)*sigma_factor,live_time=100.0,real_time=100.0,calibration={'energy':[0.0,1.0]})
    return FluxWireData(sample_id='synthetic',spectrum=spectrum,live_time=100.0,real_time=100.0,energy_calibration=[0.0,1.0],resolution=[2.0,0.0])


@pytest.mark.parametrize('method',['covell','gilmore','iec_tiered','qg'])
def test_custom_counting_sigma_is_preserved(method):
    line=GammaLine(200.0,0.5,'Present')
    low=analyze_raw_spectrum_targeted(synthetic_data(1.0),[line],background_subtract=False,counting_method=method)[0]
    high=analyze_raw_spectrum_targeted(synthetic_data(10.0),[line],background_subtract=False,counting_method=method)[0]
    assert high.net_counts == pytest.approx(low.net_counts,rel=1e-5)
    assert high.net_counts_unc == pytest.approx(10.0*low.net_counts_unc,rel=1e-4)


def test_multiplet_custom_counting_sigma_is_preserved():
    lines=[GammaLine(200.0,0.5,'Present'),GammaLine(205.0,0.5,'Present')]
    low=analyze_raw_spectrum_targeted(synthetic_data(1.0,True),lines,background_subtract=False,counting_method='iec_tiered')
    high=analyze_raw_spectrum_targeted(synthetic_data(10.0,True),lines,background_subtract=False,counting_method='iec_tiered')
    assert len(low)==len(high)==2
    for one,two in zip(low,high):
        assert two.net_counts_unc == pytest.approx(10.0*one.net_counts_unc,rel=1e-4)


def test_nearby_strong_line_does_not_satisfy_wrong_expected_identity():
    peaks=analyze_raw_spectrum_targeted(synthetic_data(),[GammaLine(205.0,0.5,'Absent')],peak_threshold=3.0,background_subtract=False,counting_method='iec_tiered')
    assert peaks == []


def test_absent_neighbor_does_not_erase_real_supported_line():
    peaks=analyze_raw_spectrum_targeted(synthetic_data(),[GammaLine(200.0,0.5,'Present'),GammaLine(203.0,0.1,'Absent')],peak_threshold=3.0,background_subtract=False,counting_method='iec_tiered')
    supported=[peak for peak in peaks if peak.isotope=='Present']
    assert len(supported)==1
    assert supported[0].energy_keV==pytest.approx(200.0,abs=0.25)
    assert supported[0].net_counts==pytest.approx(2000.0*1.2*np.sqrt(2*np.pi),rel=0.1)


def test_resolved_weak_neighbor_is_recovered_without_biasing_strong_peak():
    x=np.arange(512,dtype=float)
    counts=40.0+2000.0*np.exp(-0.5*((x-200.0)/1.2)**2)+500.0*np.exp(-0.5*((x-204.0)/1.2)**2)
    spectrum=GammaSpectrum(counts=counts,live_time=100.0,real_time=100.0,calibration={'energy':[0.0,1.0]})
    data=FluxWireData(sample_id='real_doublet',spectrum=spectrum,energy_calibration=[0.0,1.0],resolution=[2.0,0.0])
    peaks=analyze_raw_spectrum_targeted(data,[GammaLine(200.0,0.5,'Strong'),GammaLine(204.0,0.5,'Weak')],peak_threshold=3.0,background_subtract=False,counting_method='iec_tiered')
    by_isotope={peak.isotope:peak for peak in peaks}
    assert set(by_isotope)=={'Strong','Weak'}
    assert by_isotope['Strong'].net_counts==pytest.approx(2000.0*1.2*np.sqrt(2*np.pi),rel=0.01)
    assert by_isotope['Weak'].net_counts==pytest.approx(500.0*1.2*np.sqrt(2*np.pi),rel=0.01)


def test_declared_high_variance_bin_does_not_bias_peak_fit():
    data=synthetic_data(multiplet=True)
    data.spectrum.counts[201]+=1000.0
    data.spectrum.counts_uncertainty[201]=1.0e6
    peaks=analyze_raw_spectrum_targeted(data,[GammaLine(200.0,0.5,'Strong'),GammaLine(205.0,0.5,'Weak')],background_subtract=False,counting_method='iec_tiered')
    by_isotope={peak.isotope:peak for peak in peaks}
    assert set(by_isotope)=={'Strong','Weak'}
    assert by_isotope['Strong'].net_counts==pytest.approx(2000.0*1.2*np.sqrt(2*np.pi),rel=0.01)
    assert by_isotope['Weak'].net_counts==pytest.approx(1000.0*1.2*np.sqrt(2*np.pi),rel=0.01)


@pytest.mark.parametrize('method',['covell','gilmore','iec_tiered','qg'])
def test_measured_background_peak_with_insignificant_residual_is_not_sample_activity(method):
    x=np.arange(512,dtype=float)
    ambient_counts=40.0+2000.0*np.exp(-0.5*((x-200.0)/1.2)**2)
    sample_counts=ambient_counts+np.exp(-0.5*((x-200.0)/1.2)**2)
    sample=GammaSpectrum(counts=sample_counts,live_time=100.0,real_time=100.0,calibration={'energy':[0.0,1.0]})
    ambient=GammaSpectrum(counts=ambient_counts,live_time=100.0,real_time=100.0,calibration={'energy':[0.0,1.0]})
    data=FluxWireData(sample_id='ambient_only',spectrum=sample,energy_calibration=[0.0,1.0],resolution=[2.0,0.0])
    peaks=analyze_raw_spectrum_targeted(data,[GammaLine(200.0,0.5,'Ambient')],peak_threshold=3.0,background_spectrum=ambient,counting_method=method)
    assert peaks==[]


@pytest.mark.parametrize('method',['qg','iec_tiered'])
def test_background_subtracted_multiplet_keeps_raw_comparison_basis(method):
    x=np.arange(512,dtype=float)
    background=40.0+2000.0*np.exp(-0.5*((x-200.0)/1.2)**2)+1000.0*np.exp(-0.5*((x-205.0)/1.2)**2)
    counts=background+500.0*np.exp(-0.5*((x-200.0)/1.2)**2)+300.0*np.exp(-0.5*((x-205.0)/1.2)**2)
    sample=GammaSpectrum(counts=counts,live_time=100.0,real_time=100.0,calibration={'energy':[0.0,1.0]})
    ambient=GammaSpectrum(counts=background,live_time=100.0,real_time=100.0,calibration={'energy':[0.0,1.0]})
    data=FluxWireData(sample_id='multiplet_background',spectrum=sample,energy_calibration=[0.0,1.0],resolution=[2.0,0.0])
    peaks=analyze_raw_spectrum_targeted(data,[GammaLine(200.0,0.5,'Strong'),GammaLine(205.0,0.5,'Weak')],peak_threshold=3.0,background_spectrum=ambient,counting_method=method)
    by_isotope={peak.isotope:peak for peak in peaks}
    assert set(by_isotope)=={'Strong','Weak'}
    for isotope,physical_amp,raw_amp in [('Strong',500.0,2500.0),('Weak',300.0,1300.0)]:
        peak=by_isotope[isotope]
        assert peak.net_counts==pytest.approx(physical_amp*1.2*np.sqrt(2*np.pi),rel=0.01)
        assert peak.comparison_net_counts==pytest.approx(raw_amp*1.2*np.sqrt(2*np.pi),rel=0.01)
