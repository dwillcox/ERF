#include <TimeIntegration.H>

using namespace amrex;

// Compute tau_ij (m + 1/2), tau_ij (m - 1/2) where m = {i, j, k} for DNS or Smagorinsky
AMREX_GPU_DEVICE
Real ComputeStressTerm (const int &i, const int &j, const int &k,
                        const Array4<Real>& u, const Array4<Real>& v, const Array4<Real>& w,
                        const enum NextOrPrev &nextOrPrev,
                        const enum MomentumEqn &momentumEqn,
                        const enum DiffusionDir &diffDir,
                        const GpuArray<Real, AMREX_SPACEDIM>& cellSize,
                        const Array4<Real>& Ksmag,
                        const SolverChoice &solverChoice,
                        bool use_no_slip_stencil) {

    // Here, we have computed strain rate on the fly.
    // TODO: It may be better to store S11, S12 etc. at all the (m+1/2) and (m-1/2) grid points (edges) and use them here.
    Real strainRate = ComputeStrainRate(i, j, k, u, v, w, nextOrPrev, momentumEqn, diffDir, cellSize, use_no_slip_stencil);

    Real viscosity_factor = 0.0;
    //TODO: dynamic viscosity, mu, is assumed to be constant in the current implementation.
    // Future implementations may account for mu = mu(T) computed at the coordinate of interest.
    // That could be done with a new MolecDiffType
    switch (solverChoice.molec_diff_type) {
        case MolecDiffType::Constant:
        viscosity_factor += 2.0 * solverChoice.dynamicViscosity;
        break;
        case MolecDiffType::None:
        break;
        default:
        amrex::Abort("Error: Molecular diffusion/viscosity model is unrecognized");
    }

    Real turbViscInterpolated = 0.0;
    // TODO: Add Deardorff model, perhaps take advantage of turbulence model indicator
    switch (solverChoice.les_type) {
        case LESType::Smagorinsky:
        turbViscInterpolated = InterpolateTurbulentViscosity(i, j, k, u, v, w, nextOrPrev, momentumEqn, diffDir, cellSize, Ksmag);
        viscosity_factor += turbViscInterpolated;
        break;
        case LESType::None:
        break;
        default:
            amrex::Abort("Error:  LES model is unrecognized");
    }

    Real stressTerm = viscosity_factor * strainRate;
    return stressTerm;
}

AMREX_GPU_DEVICE
Real
DiffusionContributionForMom(const int &i, const int &j, const int &k,
                            const Array4<Real>& u, const Array4<Real>& v, const Array4<Real>& w,
                            const enum MomentumEqn &momentumEqn,
                            const GpuArray<Real, AMREX_SPACEDIM>& cellSize,
                            const Array4<Real>& Ksmag,
                            const SolverChoice &solverChoice,
                            const bool use_no_slip_stencil_at_lo_k,
                            const bool use_no_slip_stencil_at_hi_k)
{
    auto dx = cellSize[0], dy = cellSize[1], dz = cellSize[2];
    Real diffusionContribution = 0.0;

    switch (momentumEqn) {
        case MomentumEqn::x:
            Real tau11Next, tau11Prev, tau12Next, tau12Prev, tau13Next, tau13Prev;
            tau11Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::x, cellSize, Ksmag, solverChoice, false);
            tau11Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::x, cellSize, Ksmag, solverChoice, false);
            tau12Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::y, cellSize, Ksmag, solverChoice, false);
            tau12Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::y, cellSize, Ksmag, solverChoice, false);
            tau13Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::z, cellSize, Ksmag, solverChoice, use_no_slip_stencil_at_hi_k);
            tau13Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::z, cellSize, Ksmag, solverChoice, use_no_slip_stencil_at_lo_k);

            diffusionContribution = (tau11Next - tau11Prev) / dx  // Contribution to x-mom eqn from diffusive flux in x-dir
                                  + (tau12Next - tau12Prev) / dy  // Contribution to x-mom eqn from diffusive flux in y-dir
                                  + (tau13Next - tau13Prev) / dz; // Contribution to x-mom eqn from diffusive flux in z-dir
            break;
        case MomentumEqn::y:
            Real tau21Next, tau21Prev, tau22Next, tau22Prev, tau23Next, tau23Prev;
            tau21Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::x, cellSize, Ksmag, solverChoice, false);
            tau21Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                           DiffusionDir::x, cellSize, Ksmag, solverChoice, false);
            tau22Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::y, cellSize, Ksmag, solverChoice, false);
            tau22Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::y, cellSize, Ksmag, solverChoice, false);
            tau23Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::z, cellSize, Ksmag, solverChoice, use_no_slip_stencil_at_hi_k);
            tau23Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::z, cellSize, Ksmag, solverChoice, use_no_slip_stencil_at_lo_k);

            diffusionContribution = (tau21Next - tau21Prev) / dx  // Contribution to y-mom eqn from diffusive flux in x-dir
                                  + (tau22Next - tau22Prev) / dy  // Contribution to y-mom eqn from diffusive flux in y-dir
                                  + (tau23Next - tau23Prev) / dz; // Contribution to y-mom eqn from diffusive flux in z-dir
            break;
        case MomentumEqn::z:
            Real tau31Next, tau31Prev, tau32Next, tau32Prev, tau33Next, tau33Prev;
            tau31Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::x, cellSize, Ksmag, solverChoice, false);
            tau31Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::x, cellSize, Ksmag, solverChoice, false);
            tau32Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::y, cellSize, Ksmag, solverChoice, false);
            tau32Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::y, cellSize, Ksmag, solverChoice, false);
            tau33Next = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::next, momentumEqn,
                                          DiffusionDir::z, cellSize, Ksmag, solverChoice, false);
            tau33Prev = ComputeStressTerm(i, j, k, u, v, w, NextOrPrev::prev, momentumEqn,
                                          DiffusionDir::z, cellSize, Ksmag, solverChoice, false);

            diffusionContribution = (tau31Next - tau31Prev) / dx  // Contribution to z-mom eqn from diffusive flux in x-dir
                                  + (tau32Next - tau32Prev) / dy  // Contribution to z-mom eqn from diffusive flux in y-dir
                                  + (tau33Next - tau33Prev) / dz; // Contribution to z-mom eqn from diffusive flux in z-dir
            break;
        default:
            amrex::Abort("Error: Momentum equation is unrecognized");
    }

    return diffusionContribution;
}

AMREX_GPU_DEVICE
amrex::Real ComputeDiffusionFluxForState(const int &i, const int &j, const int &k,
                     const Array4<Real>& cell_data, const int & qty_index,
                     const amrex::Real invCellWidth,
                     const Array4<Real>& Ksmag,
                     const SolverChoice &solverChoice,
                     const enum NextOrPrev &nextOrPrev,
                     const enum Coord& coordDir)
{
  // Get indices of states to left and right of the face on which we want the flux
  const int il = i - ((coordDir == Coord::x) && (nextOrPrev == NextOrPrev::prev));
  const int ir = i + ((coordDir == Coord::x) && (nextOrPrev == NextOrPrev::next));
  const int jl = j - ((coordDir == Coord::y) && (nextOrPrev == NextOrPrev::prev));
  const int jr = j + ((coordDir == Coord::y) && (nextOrPrev == NextOrPrev::next));
  const int kl = k - ((coordDir == Coord::z) && (nextOrPrev == NextOrPrev::prev));
  const int kr = k + ((coordDir == Coord::z) && (nextOrPrev == NextOrPrev::next));

  // Get diffusion coefficients
  amrex::Real rhoAlpha_molec;
  amrex::Real Pr_or_Sc_turb_inv;
  switch(qty_index) {
  case RhoTheta_comp: // Temperature
    rhoAlpha_molec = solverChoice.rhoAlpha_T;
    Pr_or_Sc_turb_inv = solverChoice.Pr_t_inv;
    break;
  case RhoScalar_comp: // Scalar
    rhoAlpha_molec = solverChoice.rhoAlpha_C;
    Pr_or_Sc_turb_inv = solverChoice.Sc_t_inv;
    break;
  default:
    amrex::Abort("Error: Diffusion term for the data index isn't implemented");
  }

  amrex::Real rhoAlpha = 0.0;
  switch (solverChoice.molec_diff_type) {
  case MolecDiffType::Constant:
    rhoAlpha += rhoAlpha_molec;
    break;
  case MolecDiffType::None:
    break;
  default:
    amrex::Abort("Error: Molecular diffusion/viscosity model is unrecognized");
  }

  amrex::Real rhoAlpha_r = 0.0, rhoAlpha_l = 0.0;
  // TODO: Add Deardorff model, perhaps take advantage of turbulence model indicator
  switch (solverChoice.les_type) {
  case LESType::Smagorinsky:
    // Ksmag = 2*mu_t -> extra factor of 0.5 when computing rhoAlpha
    rhoAlpha_r = Ksmag(ir, jr, kr) * Pr_or_Sc_turb_inv;
    rhoAlpha_l = Ksmag(il, jl, kl) * Pr_or_Sc_turb_inv;
    rhoAlpha += 0.25*(rhoAlpha_l + rhoAlpha_r);
    break;
  case LESType::None:
    break;
  default:
    amrex::Abort("Error:  LES model is unrecognized");
  }

  // Compute the flux
  // TODO : could be more efficient to compute comp from Rho_comp before this
  amrex::Real diffusionFlux = rhoAlpha * invCellWidth *
      (cell_data(ir, jr, kr, qty_index) / cell_data(ir, jr, kr, Rho_comp)
     - cell_data(il, jl, kl, qty_index) / cell_data(il, jl, kl, Rho_comp));

  return diffusionFlux;
}

AMREX_GPU_DEVICE
Real
DiffusionContributionForState(const int &i, const int &j, const int &k,
                              const Array4<Real>& cell_data, const int & qty_index,
                              const Array4<Real>& xflux, const Array4<Real>& yflux, const Array4<Real>& zflux,
                              const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
                              const Array4<Real>& Ksmag,
                              const SolverChoice &solverChoice)
{
  const amrex::Real dx_inv = cellSizeInv[0];
  const amrex::Real dy_inv = cellSizeInv[1];
  const amrex::Real dz_inv = cellSizeInv[2];

  // TODO : could be more efficient to compute and save all fluxes before taking divergence (now all fluxes are computed 2x);
  xflux(i+1,j,k,qty_index) = ComputeDiffusionFluxForState(i, j, k, cell_data, qty_index, dx_inv, Ksmag, solverChoice, NextOrPrev::next, Coord::x);
  yflux(i,j+1,k,qty_index) = ComputeDiffusionFluxForState(i, j, k, cell_data, qty_index, dy_inv, Ksmag, solverChoice, NextOrPrev::next, Coord::y);
  zflux(i,j,k+1,qty_index) = ComputeDiffusionFluxForState(i, j, k, cell_data, qty_index, dz_inv, Ksmag, solverChoice, NextOrPrev::next, Coord::z);

  xflux(i  ,j,k,qty_index) = ComputeDiffusionFluxForState(i, j, k, cell_data, qty_index, dx_inv, Ksmag, solverChoice, NextOrPrev::prev, Coord::x);
  yflux(i,j  ,k,qty_index) = ComputeDiffusionFluxForState(i, j, k, cell_data, qty_index, dy_inv, Ksmag, solverChoice, NextOrPrev::prev, Coord::y);
  zflux(i,j,k  ,qty_index) = ComputeDiffusionFluxForState(i, j, k, cell_data, qty_index, dz_inv, Ksmag, solverChoice, NextOrPrev::prev, Coord::z);

  Real diffusionContribution =
      (xflux(i+1,j,k,qty_index) - xflux(i  ,j,k,qty_index)) * dx_inv   // Diffusive flux in x-dir
     +(yflux(i,j+1,k,qty_index) - yflux(i,j  ,k,qty_index)) * dy_inv   // Diffusive flux in y-dir
     +(zflux(i,j,k+1,qty_index) - zflux(i,j,k  ,qty_index)) * dz_inv;  // Diffusive flux in z-dir

  return diffusionContribution;
}
