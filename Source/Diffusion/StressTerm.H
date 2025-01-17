#ifndef _COMPUTE_STRESSTERM_H
#define _COMPUTE_STRESSTERM_H

#include <EddyViscosity.H>
#include <StrainRate.H>

// Compute tau_ij (m + 1/2), tau_ij (m - 1/2) where m = {i, j, k} for DNS or Smagorinsky
//    given the strain rates
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ComputeStressGivenRates (int i, int j, int k,
                         enum MomentumEqn momentumEqn,
                         enum DiffusionDir &diffDir,
                         const amrex::Array4<const amrex::Real>& K_turb,
                         const SolverChoice &solverChoice,
                         amrex::Real strainRateDeviatoric)
{
    amrex::Real mu_effective = 0.0;
    //
    // NOTE: we do not allow LES with MolecDiffType::ConstantAlpha so we don't have to worry about the
    //       density-weighting of the LES term in this case.  When using MolecDiffType::ConstantAlpha,
    //       we multiply the entire stress term by (rho / rho0_trans) in DiffusionSrcForMom
    //       (which is calling this routine)
    //
    if ( (solverChoice.molec_diff_type == MolecDiffType::Constant) ||
         (solverChoice.molec_diff_type == MolecDiffType::ConstantAlpha) ) {
        mu_effective += 2.0 * solverChoice.dynamicViscosity; // 2*mu
    }

    if ( (solverChoice.les_type == LESType::Smagorinsky) ||
         (solverChoice.les_type == LESType::Deardorff  ) ||
         (solverChoice.pbl_type == PBLType::MYNN25     )  ) {
        mu_effective += InterpolateTurbulentViscosity(i, j, k, momentumEqn, diffDir, K_turb); // 2*mu_t
    }

    amrex::Real stressTerm = mu_effective * strainRateDeviatoric; // tau_ij = mu_effective * sigma_ij
    return stressTerm;
}

// Compute tau_ij (m + 1/2), tau_ij (m - 1/2) where m = {i, j, k} for DNS or Smagorinsky
AMREX_GPU_DEVICE
#ifdef AMREX_DEBUG
inline AMREX_NO_INLINE
#else
AMREX_FORCE_INLINE
#endif
amrex::Real
ComputeStressTermWithTerrain (int i, int j, int k,
                              const amrex::Array4<const amrex::Real>& u,
                              const amrex::Array4<const amrex::Real>& v,
                              const amrex::Array4<const amrex::Real>& w,
                                    enum MomentumEqn  momentumEqn,
                                    enum DiffusionDir diffDir,
                              const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv,
                              const amrex::Array4<const amrex::Real>& K_turb,
                              const SolverChoice &solverChoice,
                              const amrex::Array4<const amrex::Real>& z_nd,
                              const amrex::Array4<const amrex::Real>& detJ,
                              const amrex::Box& domain, const amrex::BCRec* bc_ptr,
                              amrex::Real expansionRate)
{
    const auto& dom_lo = amrex::lbound(domain);
    const auto& dom_hi = amrex::ubound(domain);

    // We don't want to compute strain or expansion below the bottom surface
    int kk = std::max(k,dom_lo.z);

    amrex::Real strainRate;

    // S_ij term
    if (k <= dom_hi.z) {
        strainRate = ComputeStrainRateWithTerrain(i, j, kk, u, v, w, momentumEqn, diffDir, dxInv, z_nd, domain, bc_ptr);
    } else {
        const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dxInv_terr = {dxInv[0], dxInv[1], dxInv[2]/detJ(i,j,k)};
        strainRate = ComputeStrainRate(i, j, k, u, v, w, momentumEqn, diffDir, dxInv_terr, domain, bc_ptr);
    }

    // sigma_ij = S_ij - D_ij
    amrex::Real strainRateDeviatoric = strainRate - (1.0/3.0)*expansionRate;

    amrex::Real stressTerm = ComputeStressGivenRates(i, j, k, momentumEqn, diffDir, K_turb, solverChoice, strainRateDeviatoric);

    return stressTerm;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ComputeStressTerm (int i, int j, int k,
                   const amrex::Array4<const amrex::Real>& u,
                   const amrex::Array4<const amrex::Real>& v,
                   const amrex::Array4<const amrex::Real>& w,
                         enum MomentumEqn  momentumEqn,
                         enum DiffusionDir diffDir,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& cellSizeInv,
                   const amrex::Array4<const amrex::Real>& K_turb,
                   const SolverChoice &solverChoice,
                   const amrex::Box& domain, const amrex::BCRec* bc_ptr,
                   amrex::Real expansionRate)
{
    const auto& dom_lo = amrex::lbound(domain);

    // We don't want to compute strain or expansion below the bottom surface
    int kk = std::max(k,dom_lo.z);

    // S_ij term
    amrex::Real strainRate = ComputeStrainRate(i, j, kk, u, v, w, momentumEqn, diffDir, cellSizeInv, domain, bc_ptr);

    // sigma_ij = S_ij - D_ij
    amrex::Real strainRateDeviatoric = strainRate - (1.0/3.0)*expansionRate;

    amrex::Real stressTerm = ComputeStressGivenRates(i, j, k, momentumEqn, diffDir, K_turb, solverChoice, strainRateDeviatoric);

    return stressTerm;
}

#endif
