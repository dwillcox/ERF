#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_BC_TYPES.H>
#include <AMReX_TimeIntegrator.H>
#include <TimeIntegration.H>
#include <ERF.H>
#include <utils.H>

using namespace amrex;

void erf_advance(int level,
                 MultiFab& cons_old,  MultiFab& cons_new,
                 MultiFab& xvel_old, MultiFab& yvel_old, MultiFab& zvel_old,
                 MultiFab& xvel_new, MultiFab& yvel_new, MultiFab& zvel_new,
                 MultiFab& xmom_crse, MultiFab& ymom_crse, MultiFab& zmom_crse,
                 MultiFab& source,
                 std::array< MultiFab, AMREX_SPACEDIM>& flux,
                 const amrex::Geometry crse_geom,
                 const amrex::Geometry fine_geom,
                 const amrex::IntVect ref_ratio,
                 const amrex::Real dt, const amrex::Real time,
                 amrex::InterpFaceRegister* ifr,
                 const SolverChoice& solverChoice,
                 const amrex::Real* dptr_dens_hse,
                 const amrex::Real* dptr_pres_hse,
                 const amrex::Real* dptr_rayleigh_tau,
                 const amrex::Real* dptr_rayleigh_ubar,
                 const amrex::Real* dptr_rayleigh_vbar,
                 const amrex::Real* dptr_rayleigh_thetabar)
{
    BL_PROFILE_VAR("erf_advance()",erf_advance);

    int nvars = cons_old.nComp();

    const BoxArray& ba            = cons_old.boxArray();
    const DistributionMapping& dm = cons_old.DistributionMap();

    // **************************************************************************************
    // Temporary array that we use to store primitive advected quantities for the RHS
    // **************************************************************************************
    auto cons_to_prim = [&](const MultiFab& cons_state, MultiFab& prim_state) {
      for (MFIter mfi(cons_state,TilingIfNotGPU()); mfi.isValid(); ++mfi) {
          const Box& gbx = mfi.growntilebox(cons_state.nGrowVect());
          const Array4<const Real>& cons_arr = cons_state.array(mfi);
          const Array4<Real>& prim_arr = prim_state.array(mfi);

          amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            for (int n = 0; n < NUM_PRIM; ++n) {
              prim_arr(i,j,k,PrimTheta_comp + n) = cons_arr(i,j,k,RhoTheta_comp + n) / cons_arr(i,j,k,Rho_comp);
            }
          });
      }
    };

    MultiFab S_prim(ba, dm, NUM_PRIM, cons_old.nGrowVect());

    // **************************************************************************************
    // These are temporary arrays that we use to store the accumulation of the fluxes
    // **************************************************************************************
    std::array< MultiFab, AMREX_SPACEDIM >  advflux;
    std::array< MultiFab, AMREX_SPACEDIM > diffflux;

     advflux[0].define(convert(ba,IntVect(1,0,0)), dm, nvars, 0);
    diffflux[0].define(convert(ba,IntVect(1,0,0)), dm, nvars, 0);

     advflux[1].define(convert(ba,IntVect(0,1,0)), dm, nvars, 0);
    diffflux[1].define(convert(ba,IntVect(0,1,0)), dm, nvars, 0);

     advflux[2].define(convert(ba,IntVect(0,0,1)), dm, nvars, 0);
    diffflux[2].define(convert(ba,IntVect(0,0,1)), dm, nvars, 0);

     advflux[0].setVal(0.);
     advflux[1].setVal(0.);
     advflux[2].setVal(0.);

    diffflux[0].setVal(0.);
    diffflux[1].setVal(0.);
    diffflux[2].setVal(0.);

    // **************************************************************************************
    // Here we define state_old and state_new which are the Nvectors to be advanced
    // **************************************************************************************
    // Initial solution
    amrex::Vector<amrex::MultiFab> state_old;
    state_old.push_back(MultiFab(ba, dm, nvars, cons_old.nGrow())); // cons
    state_old.push_back(MultiFab(convert(ba,IntVect(1,0,0)), dm, 1, xvel_old.nGrow())); // xmom
    state_old.push_back(MultiFab(convert(ba,IntVect(0,1,0)), dm, 1, yvel_old.nGrow())); // ymom
    state_old.push_back(MultiFab(convert(ba,IntVect(0,0,1)), dm, 1, zvel_old.nGrow())); // zmom
    state_old.push_back(MultiFab(convert(ba,IntVect(1,0,0)), dm, nvars, 1)); // x-fluxes
    state_old.push_back(MultiFab(convert(ba,IntVect(0,1,0)), dm, nvars, 1)); // y-fluxes
    state_old.push_back(MultiFab(convert(ba,IntVect(0,0,1)), dm, nvars, 1)); // z-fluxes

    // Final solution
    amrex::Vector<amrex::MultiFab> state_new;
    state_new.push_back(MultiFab(ba, dm, nvars, cons_old.nGrow())); // cons
    state_new.push_back(MultiFab(convert(ba,IntVect(1,0,0)), dm, 1, xvel_old.nGrow())); // xmom
    state_new.push_back(MultiFab(convert(ba,IntVect(0,1,0)), dm, 1, yvel_old.nGrow())); // ymom
    state_new.push_back(MultiFab(convert(ba,IntVect(0,0,1)), dm, 1, zvel_old.nGrow())); // zmom
    state_new.push_back(MultiFab(convert(ba,IntVect(1,0,0)), dm, nvars, 1)); // x-fluxes
    state_new.push_back(MultiFab(convert(ba,IntVect(0,1,0)), dm, nvars, 1)); // y-fluxes
    state_new.push_back(MultiFab(convert(ba,IntVect(0,0,1)), dm, nvars, 1)); // z-fluxes

    // **************************************************************************************
    // Prepare the old-time data for calling the integrator
    // **************************************************************************************

    // Note that we have filled the ghost cells of cons_old and we are copying the ghost cell values
    //      so we don't need to enforce BCs again here
    MultiFab::Copy(state_old[IntVar::cons], cons_old, 0, 0, cons_old.nComp(), cons_old.nGrow());

    // Convert old velocity available on faces to old momentum on faces to be used in time integration
    // **************************************************************************************
    VelocityToMomentum(xvel_old, yvel_old, zvel_old,
                       state_old[IntVar::cons],
                       state_old[IntVar::xmom],
                       state_old[IntVar::ymom],
                       state_old[IntVar::zmom],
                       solverChoice.spatial_order,
                       xvel_old.nGrow());

    // Apply BC on old momentum data on faces before integration
    // **************************************************************************************
    state_old[IntVar::xmom].FillBoundary(fine_geom.periodicity());
    state_old[IntVar::ymom].FillBoundary(fine_geom.periodicity());
    state_old[IntVar::zmom].FillBoundary(fine_geom.periodicity());

    // **************************************************************************************
    // Initialize the fluxes to zero
    // **************************************************************************************
    state_old[IntVar::xflux].setVal(0.0_rt);
    state_old[IntVar::yflux].setVal(0.0_rt);
    state_old[IntVar::zflux].setVal(0.0_rt);

    auto interpolate_coarse_fine_faces = [&](Vector<MultiFab>& S_data) {
        if (level > 0)
        {
            amrex::Array<const MultiFab*,3> cmf_const{&xmom_crse, &ymom_crse, &zmom_crse};
            amrex::Array<MultiFab*,3> fmf{&S_data[IntVar::xmom],
                                          &S_data[IntVar::ymom],
                                          &S_data[IntVar::zmom]};

            // Interpolate from coarse faces to fine faces *only* on the coarse-fine boundary
            ifr->interp(fmf,cmf_const,0,1);

            amrex::Array<MultiFab*,3> cmf{&xmom_crse, &ymom_crse, &zmom_crse};

            int nGrow = 1;
            BoxArray fine_grids(cons_old.boxArray());

            // Interpolate from coarse faces on fine faces outside the fine region
            create_umac_grown(level, nGrow, fine_grids, crse_geom, fine_geom, cmf, fmf, ref_ratio);
        }
    };

    auto apply_bcs = [&](Vector<MultiFab>& S_data) {
        amrex::Vector<MultiFab*> state_p{&S_data[IntVar::cons],
                                         &S_data[IntVar::xmom],
                                         &S_data[IntVar::ymom],
                                         &S_data[IntVar::zmom]};

        ERF::applyBCs(fine_geom, state_p);

        MomentumToVelocity(xvel_new, yvel_new, zvel_new,
                           S_data[IntVar::cons],
                           S_data[IntVar::xmom],
                           S_data[IntVar::ymom],
                           S_data[IntVar::zmom],
                           solverChoice.spatial_order,
                           xvel_new.nGrow());

        xvel_new.FillBoundary(fine_geom.periodicity());
        yvel_new.FillBoundary(fine_geom.periodicity());
        zvel_new.FillBoundary(fine_geom.periodicity());

        // Apply BC on velocity data on faces
        // Note that the BC was already applied on momentum
        amrex::Vector<MultiFab*> vel_vars{&xvel_new, &yvel_new, &zvel_new};
        ERF::applyBCs(fine_geom, vel_vars);
    };

    interpolate_coarse_fine_faces(state_old);
    apply_bcs(state_old);
    cons_to_prim(state_old[IntVar::cons], S_prim);

    // **************************************************************************************
    // Setup the integrator
    // **************************************************************************************
    TimeIntegrator<amrex::Vector<amrex::MultiFab> > integrator(state_old);

    //Create function lambdas
    bool l_lo_z_is_dirichlet = ERF::lo_z_is_dirichlet;
    bool l_hi_z_is_dirichlet = ERF::hi_z_is_dirichlet;

    auto rhs_fun = [&](      Vector<MultiFab>& S_rhs,
                       const Vector<MultiFab>& S_data, const Real time) {
        erf_rhs(level, S_rhs, S_data, S_prim,
                xvel_new, yvel_new, zvel_new,
                source,
                advflux, diffflux,
                fine_geom, dt,
                ifr,
                solverChoice,
                l_lo_z_is_dirichlet,
                l_hi_z_is_dirichlet,
                dptr_dens_hse, dptr_pres_hse,
                dptr_rayleigh_tau, dptr_rayleigh_ubar,
                dptr_rayleigh_vbar, dptr_rayleigh_thetabar);
    };

    auto rhs_fun_fast = [&](      Vector<MultiFab>& S_rhs,
                            const Vector<MultiFab>& S_stage_data,
                            const Vector<MultiFab>& S_data, const Real time) {
        erf_fast_rhs(level, S_rhs, S_stage_data, S_data,
                     advflux, diffflux,
                     fine_geom, dt,
                     ifr,
                     solverChoice,
                     dptr_dens_hse, dptr_pres_hse,
                     dptr_rayleigh_tau, dptr_rayleigh_ubar,
                     dptr_rayleigh_vbar, dptr_rayleigh_thetabar);
    };

    // NOTE: We will need a post_update_fast_fun to update BCs at every fast RK stage

    auto post_update_fun = [&](Vector<MultiFab>& S_data, const Real time) {
        // Apply BC on updated state and momentum data
        for (auto& mfp : S_data) {
            mfp.FillBoundary(fine_geom.periodicity());
        }

        // TODO: we should interpolate coarse data in time first, so that this interplation
        // in space is at the correct time indicated by the `time` function argument.
        interpolate_coarse_fine_faces(S_data);
        apply_bcs(S_data);
        cons_to_prim(S_data[IntVar::cons], S_prim);
    };

    // define rhs and 'post update' utility function that is called after calculating
    // any state data (e.g. at RK stages or at the end of a timestep)
    integrator.set_rhs(rhs_fun);
    integrator.set_fast_rhs(rhs_fun_fast);
    integrator.set_post_update(post_update_fun);

    // **************************************************************************************
    // Integrate for a single timestep
    // **************************************************************************************
    integrator.advance(state_old, state_new, time, dt);

    // **************************************************************************************
    // Convert updated momentum to updated velocity on faces after we have taken a timestep
    // **************************************************************************************
    MomentumToVelocity(xvel_new, yvel_new, zvel_new,
                       state_new[IntVar::cons],
                       state_new[IntVar::xmom],
                       state_new[IntVar::ymom],
                       state_new[IntVar::zmom],
                       solverChoice.spatial_order, 0);

    // **************************************************************************************
    // Get the final cell centered variables after the step
    // (do this at the very end because its a swap not a copy)
    // **************************************************************************************
    std::swap(cons_new, state_new[IntVar::cons]);

    std::swap(flux[0], state_new[IntVar::xflux]);
    std::swap(flux[1], state_new[IntVar::yflux]);
    std::swap(flux[2], state_new[IntVar::zflux]);

    // One final application of internal and periodic ghost cell filling
    xvel_new.FillBoundary(fine_geom.periodicity());
    yvel_new.FillBoundary(fine_geom.periodicity());
    zvel_new.FillBoundary(fine_geom.periodicity());

    // One final application of non-periodic BCs
    amrex::Vector<MultiFab*> vars{&cons_new, &xvel_new, &yvel_new, &zvel_new};
    ERF::applyBCs(fine_geom, vars);

}
