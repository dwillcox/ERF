    auto fast_rhs_fun = [&](int fast_step,
                            Vector<MultiFab>& S_slow_rhs,
                            const Vector<MultiFab>& S_old,
                            Vector<MultiFab>& S_stage,
                            Vector<MultiFab>& S_data,
                            Vector<MultiFab>& S_scratch,
                            const Real dtau,
                            const Real inv_fac,
                            const Real old_time,
                            const Real new_time,
                            const Real old_substep_time,
                            const Real new_substep_time)
    {
        BL_PROFILE("fast_rhs_fun");
        if (verbose) amrex::Print() << "Calling fast rhs at level " << level << " with dt = " << dtau << std::endl;

        // Moving terrain
        MultiFab* z_t_pert = nullptr;
        if ( solverChoice.use_terrain &&  (solverChoice.terrain_type == 1) )
        {
            if (verbose) Print() << "Making new geometry at substep " << new_time << std::endl;

            // Make "old" fast geom -- store in z_phys_nd for convenience
            if (verbose) Print() << "Making geometry at start of substep time: " << old_substep_time << std::endl;
            init_custom_terrain(fine_geom,*z_phys_nd[level],old_time,new_time,old_substep_time);
            init_terrain_grid  (fine_geom,*z_phys_nd[level]);
            make_metrics       (fine_geom,*z_phys_nd[level], *z_phys_cc[level], *detJ_cc[level]);

            // Make "new" fast geom
            if (verbose) Print() << "Making geometry for end of substep time :" << new_substep_time << std::endl;
            init_custom_terrain(fine_geom,*z_phys_nd_new[level],old_time,new_time,new_substep_time);
            init_terrain_grid  (fine_geom,*z_phys_nd_new[level]);
            make_metrics       (fine_geom,*z_phys_nd_new[level], *z_phys_cc_new[level], *detJ_cc_new[level]);

            Real inv_dt   = 1./dtau;

            z_t_pert = new MultiFab(S_data[IntVar::zmom].boxArray(), S_data[IntVar::zmom].DistributionMap(), 1, 1);

            for (MFIter mfi(*z_t_rk[level],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                Box gbx = mfi.growntilebox(IntVect(1,1,0));

                const Array4<Real      >& z_t_arr      =  z_t_rk[level]->array(mfi);
                const Array4<Real      >& zp_t_arr     =  z_t_pert->array(mfi);

                const Array4<Real const>& z_nd_new_arr =  z_phys_nd_new[level]->const_array(mfi);
                const Array4<Real const>& z_nd_old_arr =  z_phys_nd[level]->const_array(mfi);

                // Loop over horizontal plane
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // Evaluate between RK stages assuming the geometry is linear between old and new time
                    zp_t_arr(i,j,k) = 0.25 * inv_dt * (z_nd_new_arr(i+1,j+1,k) - z_nd_old_arr(i+1,j+1,k)
                                                      +z_nd_new_arr(i  ,j+1,k) - z_nd_old_arr(  i,j+1,k)
                                                      +z_nd_new_arr(i+1,j  ,k) - z_nd_old_arr(i+1,j  ,k)
                                                      +z_nd_new_arr(i  ,j  ,k) - z_nd_old_arr(i  ,j  ,k));
                    // Convert to perturbation: z"_t(t) = z_t(t) - z_t^{RK}
                    zp_t_arr(i,j,k) -= z_t_arr(i,j,k);
                });
            } // mfi
        } // moving terrain

        if (solverChoice.use_terrain && solverChoice.terrain_type == 1) {

            // We have to call this each step since it depends on the substep time now
            make_fast_coeffs(level, fast_coeffs, S_stage, S_prim, pi_stage, fine_geom, ifr, solverChoice,
                             z_phys_nd_new[level], detJ_cc_new[level], r0, pi0, dtau, ingested_bcs);

            if (fast_step == 0) {
                // If this is the first substep we pass in S_old as the previous step's solution
                erf_fast_rhs_MT(fast_step, level, new_time, S_slow_rhs, S_old, S_stage, S_prim, pi_stage, fast_coeffs,
                                S_data, S_scratch, fine_geom, ifr, solverChoice, Omega, // z_t_pert,
                                z_phys_nd[level], z_phys_nd_new[level], detJ_cc[level], detJ_cc_new[level],
                                dtau, inv_fac, ingested_bcs);
            } else {
                // If this is not the first substep we pass in S_data as the previous step's solution
                erf_fast_rhs_MT(fast_step, level, new_time, S_slow_rhs, S_data, S_stage, S_prim, pi_stage, fast_coeffs,
                                S_data, S_scratch, fine_geom, ifr, solverChoice, Omega, // z_t_pert,
                                z_phys_nd[level], z_phys_nd_new[level], detJ_cc[level], detJ_cc_new[level],
                                dtau, inv_fac, ingested_bcs);
            }
        } else if (solverChoice.use_terrain && solverChoice.terrain_type == 0) {
            if (fast_step == 0) {

                // If this is the first substep we make the coefficients since they are based only on stage data
                make_fast_coeffs(level, fast_coeffs, S_stage, S_prim, pi_stage, fine_geom, ifr, solverChoice,
                                 z_phys_nd[level], detJ_cc[level], r0, pi0, dtau, ingested_bcs);

                // If this is the first substep we pass in S_old as the previous step's solution
                erf_fast_rhs_T(fast_step, level, new_time, S_slow_rhs, S_old, S_stage, S_prim, pi_stage, fast_coeffs,
                               S_data, S_scratch, fine_geom, ifr, solverChoice, Omega,
                               z_phys_nd[level], detJ_cc[level], dtau, inv_fac, ingested_bcs);
            } else {
                // If this is not the first substep we pass in S_data as the previous step's solution
                erf_fast_rhs_T(fast_step, level, new_time, S_slow_rhs, S_data, S_stage, S_prim, pi_stage, fast_coeffs,
                               S_data, S_scratch, fine_geom, ifr, solverChoice, Omega,
                               z_phys_nd[level], detJ_cc[level], dtau, inv_fac, ingested_bcs);
            }
        } else {
            if (fast_step == 0) {

                // If this is the first substep we make the coefficients since they are based only on stage data
                make_fast_coeffs(level, fast_coeffs, S_stage, S_prim, pi_stage, fine_geom, ifr, solverChoice,
                                 z_phys_nd[level], detJ_cc[level], r0, pi0, dtau, ingested_bcs);

                // If this is the first substep we pass in S_old as the previous step's solution
                erf_fast_rhs_N(fast_step, level, new_time, S_slow_rhs, S_old, S_stage, S_prim, pi_stage, fast_coeffs,
                               S_data, S_scratch, fine_geom, ifr, solverChoice,
                               dtau, inv_fac, ingested_bcs);
            } else {
                // If this is not the first substep we pass in S_data as the previous step's solution
                erf_fast_rhs_N(fast_step, level, new_time, S_slow_rhs, S_data, S_stage, S_prim, pi_stage, fast_coeffs,
                               S_data, S_scratch, fine_geom, ifr, solverChoice,
                               dtau, inv_fac, ingested_bcs);
            }
        }

        // Moving terrain
        if ( solverChoice.use_terrain &&  (solverChoice.terrain_type == 1) )
        {
            delete z_t_pert;
        }

        Real time_mt = new_time - 0.5*dtau;

        // Even if we update all the conserved variables we don't need to fillpatch the slow ones every acoustic substep
        int ng_cons    = 1;
        int ng_vel     = 1;
        apply_bcs(S_data, new_time, time_mt, dtau, ng_cons, ng_vel, true);

        MultiFab::Copy(S_scratch[IntVar::cons], S_data[IntVar::cons], Cons::RhoTheta, Cons::RhoTheta, 1, ng_cons);
    };
