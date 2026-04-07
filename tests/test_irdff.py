"""
Tests for IRDFF-II database access and spectrum unfolding workflow.
"""

import unittest
import numpy as np
from pathlib import Path
import pytest


# =============================================================================
# IRDFF-II Database Tests
# =============================================================================


class TestIRDFFDatabase(unittest.TestCase):
    """Tests for IRDFFDatabase class."""

    def test_database_initialization(self):
        """Test database can be initialized."""
        from fluxforge.data.irdff import IRDFFDatabase

        db = IRDFFDatabase(auto_download=False, verbose=False)
        self.assertIsNotNone(db)
        self.assertTrue(db.cache_dir.exists())

    def test_list_reactions(self):
        """Test listing available reactions."""
        from fluxforge.data.irdff import IRDFFDatabase, IRDFF_REACTIONS

        db = IRDFFDatabase(verbose=False)

        # List all reactions
        all_reactions = db.list_reactions()
        self.assertGreater(len(all_reactions), 0)

        # List by category
        thermal = db.list_reactions("thermal")
        fast = db.list_reactions("fast")

        self.assertGreater(len(thermal), 0)
        self.assertGreater(len(fast), 0)
        self.assertIn("Co-59(n,g)Co-60", thermal)
        self.assertIn("Ti-46(n,p)Sc-46", fast)

    def test_get_builtin_cross_section(self):
        """Test getting built-in cross section data."""
        from fluxforge.data.irdff import IRDFFDatabase

        db = IRDFFDatabase(verbose=False)

        # Get a thermal capture reaction
        xs = db.get_cross_section("Co-59(n,g)Co-60")
        self.assertIsNotNone(xs)
        self.assertEqual(xs.reaction, "Co-59(n,g)Co-60")
        self.assertEqual(xs.target, "Co-59")
        self.assertEqual(xs.product, "Co-60")
        self.assertGreater(len(xs.energies), 0)
        self.assertGreater(len(xs.cross_sections), 0)

        # Check thermal cross section is large
        sigma_thermal = xs.evaluate(0.0253)  # 25.3 meV
        self.assertGreater(sigma_thermal, 10)  # Should be ~37 barn

    def test_get_threshold_cross_section(self):
        """Test getting threshold reaction cross section."""
        from fluxforge.data.irdff import IRDFFDatabase

        db = IRDFFDatabase(verbose=False)

        # Get a threshold reaction
        xs = db.get_cross_section("Ti-46(n,p)Sc-46")
        self.assertIsNotNone(xs)
        self.assertGreater(xs.threshold_eV, 1e6)  # ~1.6 MeV threshold

        # Cross section below threshold should be zero
        sigma_low = xs.evaluate(1e6)  # 1 MeV (below threshold)
        self.assertLess(sigma_low, 0.01)

        # Cross section at 14 MeV should be significant
        sigma_14MeV = xs.evaluate(14e6)
        self.assertGreater(sigma_14MeV, 0.1)  # Should be ~0.3 barn

    def test_cross_section_evaluation(self):
        """Test cross section evaluation at multiple energies."""
        from fluxforge.data.irdff import IRDFFDatabase

        db = IRDFFDatabase(verbose=False)
        xs = db.get_cross_section("Ni-58(n,p)Co-58")

        # Evaluate at array of energies
        energies = np.array([1e6, 5e6, 10e6, 14e6])
        sigmas = xs.evaluate(energies)

        self.assertEqual(len(sigmas), len(energies))
        self.assertTrue(all(sigmas >= 0))

        # Cross section should increase with energy for threshold reactions
        self.assertGreater(sigmas[1], sigmas[0])  # 5 MeV > 1 MeV

    def test_group_collapse(self):
        """Test collapsing cross section to group structure."""
        from fluxforge.data.irdff import IRDFFDatabase

        db = IRDFFDatabase(verbose=False)
        xs = db.get_cross_section("Co-59(n,g)Co-60")

        # Define simple group structure
        group_edges = np.array([1e-5, 0.55, 1e5, 1e6, 20e6])  # 4 groups

        group_xs, group_unc = xs.to_group_structure(group_edges)

        self.assertEqual(len(group_xs), 4)
        self.assertEqual(len(group_unc), 4)

        # Thermal group should have highest cross section
        self.assertGreater(group_xs[0], group_xs[2])  # thermal > fast


def test_offline_mode_blocks_irdff_auto_download(tmp_path, monkeypatch):
    from fluxforge.data.irdff import IRDFFDatabase

    monkeypatch.setenv("FLUXFORGE_OFFLINE", "1")
    database = IRDFFDatabase(cache_dir=tmp_path, auto_download=True, verbose=False)

    with pytest.raises(RuntimeError, match="FLUXFORGE_OFFLINE=1"):
        database._ensure_tab_data()


class TestEnergyStructures(unittest.TestCase):
    """Tests for energy group structures."""

    def test_flux_wire_groups(self):
        """Test flux wire energy group structure."""
        from fluxforge.data.irdff import get_flux_wire_energy_groups

        edges = get_flux_wire_energy_groups()

        self.assertGreater(len(edges), 100)  # Should have many groups
        self.assertLess(edges[0], 1e-4)  # Starts in thermal region
        self.assertGreaterEqual(edges[-1], 1e7)  # Extends to high energies
        self.assertTrue(np.all(np.diff(edges) > 0))  # Monotonically increasing

    def test_activation_groups(self):
        """Test activation energy group structure."""
        from fluxforge.data.irdff import get_activation_energy_groups

        edges = get_activation_energy_groups()

        self.assertGreater(len(edges), 20)  # Should have reasonable number of groups
        self.assertTrue(np.all(np.diff(edges) > 0))  # Monotonically increasing

        # Should include Cd cutoff
        self.assertTrue(any(np.isclose(edges, 0.55, rtol=0.1)))

    def test_sand725_grid(self):
        """Test SAND-II 725-group structure."""
        from fluxforge.data.irdff import IRDFFDatabase

        db = IRDFFDatabase(verbose=False)
        edges = db.get_energy_grid("sand725")

        self.assertEqual(len(edges), 726)  # 725 groups + 1
        self.assertLess(edges[0], 1e-4)
        self.assertGreater(edges[-1], 1e7)


class TestResponseMatrix(unittest.TestCase):
    """Tests for response matrix construction."""

    def test_build_response_matrix(self):
        """Test building response matrix from reactions."""
        from fluxforge.data.irdff import (
            build_response_matrix,
            get_flux_wire_energy_groups,
        )

        reactions = [
            "Ti-46(n,p)Sc-46",
            "Ni-58(n,p)Co-58",
            "Co-59(n,g)Co-60",
        ]

        edges = get_flux_wire_energy_groups()
        response, valid_reactions, uncertainties = build_response_matrix(
            reactions=reactions,
            energy_edges=edges,
            verbose=False,
        )

        self.assertEqual(response.shape[0], len(reactions))
        self.assertEqual(response.shape[1], len(edges) - 1)
        self.assertEqual(len(valid_reactions), len(reactions))

        # Thermal reaction should have high cross section at low energy
        co_idx = valid_reactions.index("Co-59(n,g)Co-60")
        self.assertGreater(
            response[co_idx, 0], response[co_idx, -1]
        )  # Higher at thermal


# =============================================================================
# Spectrum Unfolding Tests
# =============================================================================


class TestSpectrumUnfolder(unittest.TestCase):
    """Tests for SpectrumUnfolder class."""

    def test_unfolder_initialization(self):
        """Test unfolder can be initialized."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        unfolder = SpectrumUnfolder(verbose=False)
        self.assertIsNotNone(unfolder)
        self.assertGreater(unfolder.n_groups, 0)

    def test_add_reactions(self):
        """Test adding reactions to unfolder."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        unfolder = SpectrumUnfolder(verbose=False)

        unfolder.add_reaction(
            reaction="Ti-46(n,p)Sc-46",
            activity_Bq=1.23e5,
            uncertainty_Bq=1.23e3,
        )

        self.assertEqual(len(unfolder.measurements), 1)
        self.assertEqual(unfolder.measurements[0].reaction, "Ti-46(n,p)Sc-46")

    def test_simple_unfold(self):
        """Test simple unfolding with synthetic data."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        unfolder = SpectrumUnfolder(
            energy_structure="flux_wire",
            verbose=False,
        )

        # Add some reactions with synthetic activities
        unfolder.add_reaction("Ti-46(n,p)Sc-46", activity_Bq=1e5, uncertainty_Bq=1e4)
        unfolder.add_reaction("Ni-58(n,p)Co-58", activity_Bq=5e5, uncertainty_Bq=5e4)
        unfolder.add_reaction("Co-59(n,g)Co-60", activity_Bq=1e4, uncertainty_Bq=1e3)

        result = unfolder.unfold(method="GRAVEL", max_iterations=100)

        self.assertIsNotNone(result)
        self.assertEqual(len(result.flux), unfolder.n_groups)
        self.assertGreaterEqual(result.chi_squared, 0)
        self.assertEqual(result.method, "GRAVEL")

    def test_support_mask_freezes_weak_bins_to_initial_guess(self):
        """Support masking should leave weakly contributing bins fixed to the prior."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        energy_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        unfolder = SpectrumUnfolder(custom_energy_edges=energy_edges, verbose=False)
        unfolder.add_reaction(
            "Ti-46(n,p)Sc-46",
            activity_Bq=1.0,
            uncertainty_Bq=0.1,
            rate_per_atom=1.0,
        )
        unfolder.add_reaction(
            "Ni-58(n,p)Co-58",
            activity_Bq=0.5,
            uncertainty_Bq=0.05,
            rate_per_atom=0.5,
        )

        prior_flux = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=float)
        unfolder.set_initial_guess(prior_flux, source="synthetic prior")
        unfolder._response_matrix = np.array(
            [
                [0.9, 0.1, 0.0, 1e-8, 1e-8],
                [0.0, 0.2, 0.8, 1e-8, 1e-8],
            ],
            dtype=float,
        )
        unfolder._reaction_list = ["Ti-46(n,p)Sc-46", "Ni-58(n,p)Co-58"]
        unfolder._response_unc = np.zeros_like(unfolder._response_matrix)

        result = unfolder.unfold(
            method="MLEM",
            max_iterations=25,
            tolerance=1e-6,
            support_threshold=1e-3,
            support_metric="prior_contribution",
        )

        assert result.metadata["support_mask_applied"] is True
        assert result.metadata["support_mask_active_bins"] == 3
        np.testing.assert_allclose(result.flux[-2:], prior_flux[-2:])

    def test_prior_shape_basis_preserves_prior_partition_and_reconstructs_native_flux(self):
        """A coarse prior-shaped basis should expand back to the native grid."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        energy_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        unfolder = SpectrumUnfolder(custom_energy_edges=energy_edges, verbose=False)
        prior_flux = np.array([10.0, 6.0, 3.0, 1.0], dtype=float)
        unfolder.set_initial_guess(prior_flux, source="synthetic prior")
        unfolder.add_reaction(
            "Ti-46(n,p)Sc-46",
            activity_Bq=20.0,
            uncertainty_Bq=0.1,
            rate_per_atom=20.0,
        )
        unfolder._response_matrix = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float)
        unfolder._reaction_list = ["Ti-46(n,p)Sc-46"]
        unfolder._response_unc = np.zeros_like(unfolder._response_matrix)

        result = unfolder.unfold(
            method="MLEM",
            max_iterations=10,
            tolerance=1e-8,
            chi2_tolerance=1e-8,
            basis_edges=np.array([1.0, 3.0, 5.0], dtype=float),
        )

        np.testing.assert_allclose(result.flux, prior_flux, rtol=1e-8, atol=1e-8)
        assert result.metadata["basis_mode"] == "prior_shape"
        assert result.metadata["basis_groups"] == 2
        assert result.metadata["basis_edges"] == [1.0, 3.0, 5.0]

    def test_prior_shape_basis_solves_on_reduced_dimension(self):
        """Few-channel basis solving should fit coarse scaling factors and expand them."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        energy_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        unfolder = SpectrumUnfolder(custom_energy_edges=energy_edges, verbose=False)
        prior_flux = np.array([2.0, 2.0, 1.0, 1.0], dtype=float)
        unfolder.set_initial_guess(prior_flux, source="synthetic prior")
        unfolder.add_reaction(
            "Ti-46(n,p)Sc-46",
            activity_Bq=8.0,
            uncertainty_Bq=0.1,
            rate_per_atom=8.0,
        )
        unfolder.add_reaction(
            "Ni-58(n,p)Co-58",
            activity_Bq=1.0,
            uncertainty_Bq=0.1,
            rate_per_atom=1.0,
        )
        unfolder._response_matrix = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        unfolder._reaction_list = ["Ti-46(n,p)Sc-46", "Ni-58(n,p)Co-58"]
        unfolder._response_unc = np.zeros_like(unfolder._response_matrix)

        result = unfolder.unfold(
            method="MLEM",
            max_iterations=200,
            tolerance=1e-10,
            chi2_tolerance=1e-10,
            basis_edges=np.array([1.0, 3.0, 5.0], dtype=float),
        )

        expected_flux = np.array([4.0, 4.0, 0.5, 0.5], dtype=float)
        np.testing.assert_allclose(result.flux, expected_flux, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(result.predicted_rates, np.array([8.0, 1.0]), rtol=1e-6)

    def test_duplicate_reaction_rows_can_be_aggregated_before_unfolding(self):
        """Replicate wires with identical response rows should aggregate cleanly."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        energy_edges = np.array([1.0, 2.0, 3.0], dtype=float)
        unfolder = SpectrumUnfolder(custom_energy_edges=energy_edges, verbose=False)
        unfolder.set_initial_guess(np.array([1.0, 1.0], dtype=float), source="synthetic prior")
        unfolder.add_reaction(
            "Ti-46(n,p)Sc-46",
            activity_Bq=10.0,
            uncertainty_Bq=1.0,
            rate_per_atom=1.0,
        )
        unfolder.add_reaction(
            "Ti-46(n,p)Sc-46",
            activity_Bq=10.0,
            uncertainty_Bq=2.0,
            rate_per_atom=1.4,
        )
        unfolder.add_reaction(
            "Ni-58(n,p)Co-58",
            activity_Bq=10.0,
            uncertainty_Bq=1.0,
            rate_per_atom=0.5,
        )
        unfolder._response_matrix = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )
        unfolder._reaction_list = [
            "Ti-46(n,p)Sc-46",
            "Ti-46(n,p)Sc-46",
            "Ni-58(n,p)Co-58",
        ]
        unfolder._response_unc = np.zeros_like(unfolder._response_matrix)

        result = unfolder.unfold(
            method="MLEM",
            max_iterations=50,
            tolerance=1e-12,
            chi2_tolerance=1e-12,
            aggregate_duplicate_reactions=True,
        )

        expected_rate = (1.0 / 0.1**2 + 1.4 / 0.28**2) / (1.0 / 0.1**2 + 1.0 / 0.28**2)
        self.assertEqual(result.reactions_used, ["Ti-46(n,p)Sc-46", "Ni-58(n,p)Co-58"])
        self.assertTrue(result.metadata["duplicate_reaction_aggregation_applied"])
        self.assertEqual(result.metadata["duplicate_reaction_original_rows"], 3)
        self.assertEqual(result.metadata["duplicate_reaction_aggregated_rows"], 2)
        self.assertEqual(result.metadata["duplicate_reaction_counts"]["Ti-46(n,p)Sc-46"], 2)
        self.assertTrue(np.isclose(result.measured_rates[0], expected_rate))
        np.testing.assert_allclose(result.predicted_rates, result.measured_rates, rtol=1e-6)

    def test_quick_unfold(self):
        """Test quick_unfold convenience function."""
        from fluxforge.workflows.spectrum_unfolding import quick_unfold

        reactions = {
            "Ti-46(n,p)Sc-46": 1e5,
            "Ni-58(n,p)Co-58": 5e5,
            "Co-59(n,g)Co-60": 1e4,
        }

        result = quick_unfold(reactions, verbose=False)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.flux), 0)
        self.assertGreater(len(result.reactions_used), 0)


class TestFluxWireMeasurement(unittest.TestCase):
    """Tests for FluxWireMeasurement dataclass."""

    def test_measurement_creation(self):
        """Test creating a measurement."""
        from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement

        meas = FluxWireMeasurement(
            reaction="Ti-46(n,p)Sc-46",
            activity_Bq=1.23e5,
            uncertainty_Bq=1.23e3,
            saturation_factor=0.95,
            decay_factor=0.90,
        )

        self.assertEqual(meas.reaction, "Ti-46(n,p)Sc-46")
        self.assertEqual(meas.activity_Bq, 1.23e5)
        self.assertTrue(np.isclose(meas.relative_uncertainty, 0.01))

    def test_reaction_rate_calculation(self):
        """Test reaction rate per atom calculation."""
        from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement

        meas = FluxWireMeasurement(
            reaction="Sc-45(n,g)Sc-46",
            activity_Bq=1e5,
            saturation_factor=0.5,
            decay_factor=0.8,
            sample_mass_g=1.0e-3,
            isotope_abundance=1.0,
        )

        rate = meas.reaction_rate_per_atom
        n_atoms = (1.0e-3 / 44.955908) * 6.02214076e23
        expected = 1e5 / (n_atoms * 0.5 * 0.8)

        self.assertTrue(np.isclose(rate, expected))

    def test_reaction_rate_uses_timing_metadata(self):
        """Test saturation and decay corrections derived from irradiation metadata."""
        from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement

        meas = FluxWireMeasurement(
            reaction="Cu-63(n,g)Cu-64",
            activity_Bq=2.5e4,
            sample_mass_g=1.3748e-3,
            irradiation_time=7200.0,
            cooling_time=357.0,
        )

        lam = np.log(2.0) / 45720.0
        saturation = 1.0 - np.exp(-lam * 7200.0)
        decay = np.exp(-lam * 357.0)
        n_atoms = (1.3748e-3 / 63.546) * 6.02214076e23 * 0.6917
        expected = 2.5e4 / (n_atoms * saturation * decay)

        self.assertTrue(np.isclose(meas.effective_saturation_factor, saturation))
        self.assertTrue(np.isclose(meas.effective_decay_factor, decay))
        self.assertTrue(np.isclose(meas.reaction_rate_per_atom, expected))

    def test_reaction_rate_accepts_pre_normalized_override(self):
        """Test direct reaction-rate inputs bypass activity normalization."""
        from fluxforge.workflows.spectrum_unfolding import FluxWireMeasurement

        meas = FluxWireMeasurement(
            reaction="Ni-58(n,p)Co-58",
            activity_Bq=1.0,
            sample_mass_g=1.0e-3,
            irradiation_time=7200.0,
            cooling_time=3600.0,
            rate_per_atom=2.5e-13,
        )

        self.assertTrue(np.isclose(meas.reaction_rate_per_atom, 2.5e-13))


class TestUnfoldingResult(unittest.TestCase):
    """Tests for UnfoldingResult dataclass."""

    def test_result_properties(self):
        """Test result derived properties."""
        from fluxforge.workflows.spectrum_unfolding import UnfoldingResult

        edges = np.geomspace(1e-5, 20e6, 101)
        flux = np.random.rand(100) * 1e10

        result = UnfoldingResult(
            energy_edges=edges,
            flux=flux,
            flux_uncertainty=flux * 0.1,
            chi_squared=1.5,
            iterations=50,
            converged=True,
            method="GRAVEL",
        )

        self.assertEqual(result.n_groups, 100)
        self.assertEqual(len(result.energy_midpoints), 100)
        self.assertEqual(len(result.energy_widths), 100)
        self.assertGreater(result.integral_flux, 0)

    def test_lethargy_conversion(self):
        """Test conversion to lethargy representation."""
        from fluxforge.workflows.spectrum_unfolding import UnfoldingResult

        edges = np.geomspace(1e-5, 20e6, 101)
        flux = np.ones(100) * 1e10

        result = UnfoldingResult(energy_edges=edges, flux=flux)

        lethargy, flux_per_lethargy = result.to_lethargy()

        self.assertEqual(len(lethargy), 100)
        self.assertEqual(len(flux_per_lethargy), 100)
        self.assertTrue(np.all(lethargy >= 0))


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow."""

    def test_full_unfolding_workflow(self):
        """Test complete unfolding workflow."""
        from fluxforge.data.irdff import IRDFFDatabase, get_flux_wire_energy_groups
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        # 1. Get IRDFF-II cross sections
        db = IRDFFDatabase(verbose=False)
        xs_ti = db.get_cross_section("Ti-46(n,p)Sc-46")
        xs_ni = db.get_cross_section("Ni-58(n,p)Co-58")

        self.assertIsNotNone(xs_ti)
        self.assertIsNotNone(xs_ni)

        # 2. Get energy structure
        edges = get_flux_wire_energy_groups()
        self.assertGreater(len(edges), 100)

        # 3. Create unfolder and add measurements
        unfolder = SpectrumUnfolder(
            custom_energy_edges=edges,
            verbose=False,
        )

        unfolder.add_reaction("Ti-46(n,p)Sc-46", activity_Bq=1e5, uncertainty_Bq=1e4)
        unfolder.add_reaction("Ni-58(n,p)Co-58", activity_Bq=3e5, uncertainty_Bq=3e4)
        unfolder.add_reaction("Co-59(n,g)Co-60", activity_Bq=5e3, uncertainty_Bq=5e2)

        # 4. Run unfolding
        result = unfolder.unfold(method="GRAVEL", max_iterations=200)

        self.assertTrue(result.converged or result.iterations == 200)
        self.assertEqual(len(result.flux), len(edges) - 1)
        self.assertGreaterEqual(result.chi_squared, 0)

    def test_mlem_unfolding(self):
        """Test MLEM unfolding method."""
        from fluxforge.workflows.spectrum_unfolding import SpectrumUnfolder

        unfolder = SpectrumUnfolder(
            energy_structure="activation",
            verbose=False,
        )

        unfolder.add_reaction("Ti-46(n,p)Sc-46", activity_Bq=1e5, uncertainty_Bq=1e4)
        unfolder.add_reaction("Ni-58(n,p)Co-58", activity_Bq=3e5, uncertainty_Bq=3e4)

        result = unfolder.unfold(method="MLEM", max_iterations=100)

        self.assertEqual(result.method, "MLEM")
        self.assertGreater(len(result.flux), 0)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    unittest.main()
