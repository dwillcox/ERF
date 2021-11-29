/** \file EddyViscosity.cpp */
#include <TimeIntegration.H>

using namespace amrex;

/** Compute Eddy Viscosity */
//AMREX_GPU_DEVICE
void ComputeTurbulentViscosity(const MultiFab& xvel, const MultiFab& yvel, const MultiFab& zvel,
                               const MultiFab& cons_in, MultiFab& eddyViscosity,
                               const GpuArray<Real, AMREX_SPACEDIM>& cellSize,
                               const SolverChoice& solverChoice,
                               const bool& use_no_slip_stencil_at_lo_k, const int& klo,
                               const bool& use_no_slip_stencil_at_hi_k, const int& khi)
{
    const Real cellVol = cellSize[0] * cellSize[1] * cellSize[2];
    Real Cs = solverChoice.Cs;
    Real CsDeltaSqr = Cs*Cs * std::pow(cellVol, 2.0/3.0);

    for ( MFIter mfi(eddyViscosity,TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box &bx = mfi.tilebox();

        const Array4<Real const > &cell_data = cons_in.array(mfi);
        const Array4<Real> &K = eddyViscosity.array(mfi);

        const Array4<Real const> &u = xvel.array(mfi);
        const Array4<Real const> &v = yvel.array(mfi);
        const Array4<Real const> &w = zvel.array(mfi);

        amrex::ParallelFor(bx, eddyViscosity.nComp(),[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            Real S11 = ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::next, MomentumEqn::x, DiffusionDir::x, cellSize, false);
            Real S22 = ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::next, MomentumEqn::y, DiffusionDir::y, cellSize, false);
            Real S33 = ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::next, MomentumEqn::z, DiffusionDir::z, cellSize, false);

            Real S12 = 0.25* (
                      ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::prev, MomentumEqn::x, DiffusionDir::y, cellSize, false)
                    + ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::next, MomentumEqn::x, DiffusionDir::y, cellSize, false)
                    + ComputeStrainRate(i+1, j, k, u, v, w, NextOrPrev::prev, MomentumEqn::x, DiffusionDir::y, cellSize, false)
                    + ComputeStrainRate(i+1, j, k, u, v, w, NextOrPrev::next, MomentumEqn::x, DiffusionDir::y, cellSize, false)
                    );

            bool use_no_slip_stencil_prev = (use_no_slip_stencil_at_lo_k && (k == klo));
            bool use_no_slip_stencil_next = (use_no_slip_stencil_at_hi_k && (k == khi));

            Real S13 = 0.25* (
                      ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::prev, MomentumEqn::x, DiffusionDir::z, cellSize, use_no_slip_stencil_prev)
                    + ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::next, MomentumEqn::x, DiffusionDir::z, cellSize, use_no_slip_stencil_next)
                    + ComputeStrainRate(i+1, j, k, u, v, w, NextOrPrev::prev, MomentumEqn::x, DiffusionDir::z, cellSize, use_no_slip_stencil_prev)
                    + ComputeStrainRate(i+1, j, k, u, v, w, NextOrPrev::next, MomentumEqn::x, DiffusionDir::z, cellSize, use_no_slip_stencil_next)
                    );

            Real S23 = 0.25* (
                      ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::prev, MomentumEqn::y, DiffusionDir::z, cellSize, use_no_slip_stencil_prev)
                    + ComputeStrainRate(i, j, k, u, v, w, NextOrPrev::next, MomentumEqn::y, DiffusionDir::z, cellSize, use_no_slip_stencil_next)
                    + ComputeStrainRate(i, j+1, k, u, v, w, NextOrPrev::prev, MomentumEqn::y, DiffusionDir::z, cellSize, use_no_slip_stencil_prev)
                    + ComputeStrainRate(i, j+1, k, u, v, w, NextOrPrev::next, MomentumEqn::y, DiffusionDir::z, cellSize, use_no_slip_stencil_next)
                    );

            Real SmnSmn = S11*S11 + S22*S22 + S33*S33 + 2.0*S12*S12 + 2.0*S13*S13 + 2.0*S23*S23;
            // TODO: Check sign in the below equation
            K(i, j, k, 0) = 2.0 * CsDeltaSqr * cell_data(i, j, k, Rho_comp) * std::sqrt(2.0*SmnSmn);
        });

    } //mfi
} // function call

/// Compute Ksmag (i-1/2, j+1/2, k) etc given Ksmag (i, j, k) is known
// Note: This should be at edges for momEqnDir != diffDir, cell centers otherwise
AMREX_GPU_DEVICE
Real
InterpolateTurbulentViscosity(const int &i, const int &j, const int &k,
                              const Array4<Real>& u, const Array4<Real>& v, const Array4<Real>& w,
                              const enum NextOrPrev &nextOrPrev,
                              const enum MomentumEqn &momentumEqn,
                              const enum DiffusionDir &diffDir,
                              const GpuArray<Real, AMREX_SPACEDIM>& cellSize,
                              const Array4<Real>& Ksmag) {
  // Assuming we already have 'Ksmag' computed for all (i, j, k)
  Real turbViscInterpolated = 1.0;

  switch (momentumEqn) {
  case MomentumEqn::x: // Reference face is x-face index (i, j, k)
    switch (diffDir) {
    case DiffusionDir::x:
      if (nextOrPrev == NextOrPrev::next)    // K (i  , j, k) needed to obtain tau11 (i+1/2)
        turbViscInterpolated = Ksmag(i, j, k);
      else // nextOrPrev == NextOrPrev::prev // K (i-1, j, k) needed to obtain tau11 (i-1/2)
        turbViscInterpolated = Ksmag(i-1, j, k);
      break;
    case DiffusionDir::y:
      if (nextOrPrev == NextOrPrev::next)    // K (i-1/2, j+1/2, k) needed to obtain tau12 (j+1/2)
        turbViscInterpolated = 0.25*( Ksmag(i-1, j, k) + Ksmag(i, j, k) + Ksmag(i-1, j+1, k) + Ksmag(i, j+1, k) );
      else // nextOrPrev == NextOrPrev::prev // K (i-1/2, j-1/2, k) needed to obtain tau12 (j-1/2)
        turbViscInterpolated = 0.25*( Ksmag(i-1, j, k) + Ksmag(i, j, k) + Ksmag(i-1, j-1, k) + Ksmag(i, j-1, k) );
      break;
    case DiffusionDir::z:
      if (nextOrPrev == NextOrPrev::next)    // K (i-1/2, j, k+1/2) needed to obtain tau13 (k+1/2)
        turbViscInterpolated = 0.25*( Ksmag(i-1, j, k) + Ksmag(i, j, k) + Ksmag(i-1, j, k+1) + Ksmag(i, j, k+1) );
      else // nextOrPrev == NextOrPrev::prev // K (i-1/2, j, k-1/2) needed to obtain tau13 (k-1/2)
        turbViscInterpolated = 0.25*( Ksmag(i-1, j, k) + Ksmag(i, j, k) + Ksmag(i-1, j, k-1) + Ksmag(i, j, k-1) );
      break;
    default:
      amrex::Abort("Error: Diffusion direction is unrecognized");
    }
    break;
  case MomentumEqn::y: // Reference face is y-face index (i, j, k)
    switch (diffDir) {
    case DiffusionDir::x:
      if (nextOrPrev == NextOrPrev::next)    // K (i+1/2, j-1/2, k) needed to obtain tau21 (i+1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j-1, k) + Ksmag(i, j, k) + Ksmag(i+1, j-1, k) + Ksmag(i+1, j, k) );
      else // nextOrPrev == NextOrPrev::prev // K (i-1/2, j-1/2, k) needed to obtain tau21 (i-1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j-1, k) + Ksmag(i, j, k) + Ksmag(i-1, j-1, k) + Ksmag(i-1, j, k) );
      break;
    case DiffusionDir::y:
      if (nextOrPrev == NextOrPrev::next)    // K (i, j  , k) needed to obtain tau22 (j+1/2)
        turbViscInterpolated = Ksmag(i, j, k);
      else // nextOrPrev == NextOrPrev::prev // K (i, j-1, k) needed to obtain tau22 (j-1/2)
        turbViscInterpolated = Ksmag(i, j-1, k);
      break;
    case DiffusionDir::z:
      if (nextOrPrev == NextOrPrev::next)    // K (i, j-1/2, k+1/2) needed to obtain tau23 (k+1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j-1, k) + Ksmag(i, j, k) + Ksmag(i, j-1, k+1) + Ksmag(i, j, k+1) );
      else // nextOrPrev == NextOrPrev::prev // K (i, j-1/2, k-1/2) needed to obtain tau23 (k-1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j-1, k) + Ksmag(i, j, k) + Ksmag(i, j-1, k-1) + Ksmag(i, j, k-1) );
      break;
    default:
      amrex::Abort("Error: Diffusion direction is unrecognized");
    }
    break;
  case MomentumEqn::z: // Reference face is z-face index (i, j, k)
    switch (diffDir) {
    case DiffusionDir::x:
      if (nextOrPrev == NextOrPrev::next)    // K (i+1/2, j, k-1/2) needed to obtain tau31 (i+1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j, k-1) + Ksmag(i, j, k) + Ksmag(i+1, j, k-1) + Ksmag(i+1, j, k) );
      else // nextOrPrev == NextOrPrev::prev // K (i-1/2, j, k-1/2) needed to obtain tau31 (i-1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j, k-1) + Ksmag(i, j, k) + Ksmag(i-1, j, k-1) + Ksmag(i-1, j, k) );
      break;
    case DiffusionDir::y:
      if (nextOrPrev == NextOrPrev::next)    // K (i, j+1/2, k-1/2) needed to obtain tau32 (j+1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j, k-1) + Ksmag(i, j, k) + Ksmag(i, j+1, k-1) + Ksmag(i, j+1, k) );
      else // nextOrPrev == NextOrPrev::prev // K (i, j-1/2, k-1/2) needed to obtain tau32 (j-1/2)
        turbViscInterpolated = 0.25*( Ksmag(i, j, k-1) + Ksmag(i, j, k) + Ksmag(i, j-1, k-1) + Ksmag(i, j-1, k) );
      break;
    case DiffusionDir::z:
      if (nextOrPrev == NextOrPrev::next)    // K (i, j, k  ) needed to obtain tau33 (k+1/2)
        turbViscInterpolated = Ksmag(i, j, k);
      else // nextOrPrev == NextOrPrev::prev // K (i, j, k-1) needed to obtain tau33 (k-1/2)
        turbViscInterpolated = Ksmag(i, j, k-1);
      break;
    default:
      amrex::Abort("Error: Diffusion direction is unrecognized");
    }
    break;
  default:
    amrex::Abort("Error: Momentum equation is unrecognized");
  }

  return turbViscInterpolated;
}
