import pytest
from fluxforge.analysis.astm_e3376 import analyze_astm_e3376_plan

def test_astm_e3376_calibration_mode():
    plan = {
        "measurements": [
            {
                "measurement_id": "test_cal",
                "gross_area": 1000.0,
                "roi_channels": 10.0,
                "continuum_channels": 5.0,
                "continuum_low": 20.0,
                "continuum_high": 30.0,
                "live_time_s": 100.0,
                "emission_rate": 500.0 # Bq * probability
            }
        ]
    }
    result = analyze_astm_e3376_plan(plan)
    out = result["outputs"][0]
    
    # B = (10 / (2 * 5)) * (20 + 30) = 1.0 * 50 = 50.0
    assert out["continuum"] == pytest.approx(50.0)
    
    # N_A = 1000 - 50 = 950
    assert out["net_peak_area"] == pytest.approx(950.0)
    
    # N_p = 950 / 100 = 9.5 c/s
    assert out["net_count_rate"] == pytest.approx(9.5)
    
    # E_f = 9.5 / 500 = 0.019
    assert out["efficiency"] == pytest.approx(0.019)

def test_astm_e3376_measurement_mode():
    plan = {
        "measurements": [
            {
                "measurement_id": "test_meas",
                "gross_area": 2000.0,
                "roi_channels": 10.0,
                "continuum_channels": 5.0,
                "continuum_low": 20.0,
                "continuum_high": 30.0,
                "live_time_s": 50.0,
                "efficiency": 0.019,
                "gamma_probability": 0.95
            }
        ]
    }
    result = analyze_astm_e3376_plan(plan)
    out = result["outputs"][0]
    
    # B = 50.0
    # N_A = 2000 - 50 = 1950
    # N_p = 1950 / 50 = 39.0 c/s
    assert out["net_count_rate"] == pytest.approx(39.0)
    
    # N_R = N_p / E_f = 39 / 0.019 = 2052.6315
    assert out["emission_rate"] == pytest.approx(2052.6315789)
    
    # A = N_R / P_gamma = 2052.6315 / 0.95 = 2160.6648
    assert out["transmutation_rate"] == pytest.approx(2160.6648199)

def test_astm_e3376_edge_cases():
    plan = {
        "measurements": [
            {
                "measurement_id": "div_zero",
                "gross_area": 100.0,
                "roi_channels": 10.0,
                "continuum_channels": 0.0, # should not divide by zero
                "continuum_low": 10.0,
                "continuum_high": 10.0,
                "live_time_s": 0.0, # should not divide by zero
                "emission_rate": 0.0 # should not divide by zero
            }
        ]
    }
    result = analyze_astm_e3376_plan(plan)
    out = result["outputs"][0]
    assert out["continuum"] == 0.0
    assert out["net_peak_area"] == 100.0
    assert out["net_count_rate"] == 0.0
    assert out["efficiency"] == 0.0
