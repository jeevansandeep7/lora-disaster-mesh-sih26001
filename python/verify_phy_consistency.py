"""
verify_phy_consistency.py — verifies directional consistency between Python's
mesh path-loss SNR model and the MATLAB link-quality PDR dataset curves.

Usage:
    python verify_phy_consistency.py
"""

import numpy as np
import matplotlib.pyplot as plt
from lora_mesh.link_quality import LinkQualityModel
from lora_mesh.mesh import generate_mesh


def main():
    print("=== Verifying PHY Link Quality Model Consistency ===")
    lqm = LinkQualityModel()
    print(f"Dataset status: {'MATLAB CSV Present' if lqm.available else 'Analytic Fallback Active'}")

    snr_range = np.linspace(-10, 15, 50)
    pdr_none = [lqm.pdr(snr, fading="none", collision=False) for snr in snr_range]
    pdr_rician = [lqm.pdr(snr, fading="rician", collision=False) for snr in snr_range]
    pdr_rayleigh = [lqm.pdr(snr, fading="rayleigh", collision=False) for snr in snr_range]
    pdr_rician_coll = [lqm.pdr(snr, fading="rician", collision=True) for snr in snr_range]

    # Verify directional properties
    assert pdr_none[25] >= pdr_rician[25], "PDR for 'none' fading should be >= Rician fading"
    assert pdr_rician[25] >= pdr_rayleigh[25], "PDR for Rician fading should be >= Rayleigh fading"
    assert pdr_rician[25] >= pdr_rician_coll[25], "Collision should degrade PDR"

    print("Directional consistency checks passed successfully!")

    # Sample mesh distances and SNR
    G = generate_mesh(n_nodes=20, seed=42)
    mesh_snrs = [G.edges[u, v]["snr_db"] for u, v in G.edges]
    mesh_pdrs = [lqm.pdr(snr, "rician", False) for snr in mesh_snrs]

    print(f"Generated mesh with {len(G.edges)} links.")
    print(f"  SNR range in mesh: {min(mesh_snrs):.2f} dB to {max(mesh_snrs):.2f} dB")
    print(f"  PDR range in mesh: {min(mesh_pdrs):.2f} to {max(mesh_pdrs):.2f}")


if __name__ == "__main__":
    main()
