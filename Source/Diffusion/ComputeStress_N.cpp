#include <Diffusion.H>

using namespace amrex;

void
ComputeStressConsVisc_N(Box& bxcc, Box& tbxxy, Box& tbxxz, Box& tbxyz, Real mu_eff,
                        const Array4<const Real>& u, const Array4<const Real>& v, const Array4<const Real>& w,
                        Array4<Real>& tau11, Array4<Real>& tau22, Array4<Real>& tau33,
                        Array4<Real>& tau12, Array4<Real>& tau13, Array4<Real>& tau23,
                        const Array4<const Real>& er_arr,
                        const BCRec* bc_ptr, const GpuArray<Real, AMREX_SPACEDIM>& dxInv)
{
    Real OneThird   = (1./3.);

    // Dirichlet on left or right plane
    bool xl_v_dir = ( (bc_ptr[BCVars::yvel_bc].lo(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].lo(0) == ERFBCType::ext_dir_ingested) );
    bool xh_v_dir = ( (bc_ptr[BCVars::yvel_bc].hi(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].hi(0) == ERFBCType::ext_dir_ingested) );

    bool xl_w_dir = ( (bc_ptr[BCVars::zvel_bc].lo(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].lo(0) == ERFBCType::ext_dir_ingested) );
    bool xh_w_dir = ( (bc_ptr[BCVars::zvel_bc].hi(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].hi(0) == ERFBCType::ext_dir_ingested) );

    // Dirichlet on front or back plane
    bool yl_u_dir = ( (bc_ptr[BCVars::xvel_bc].lo(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].lo(1) == ERFBCType::ext_dir_ingested) );
    bool yh_u_dir = ( (bc_ptr[BCVars::xvel_bc].hi(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].hi(1) == ERFBCType::ext_dir_ingested) );

    bool yl_w_dir = ( (bc_ptr[BCVars::zvel_bc].lo(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].lo(1) == ERFBCType::ext_dir_ingested) );
    bool yh_w_dir = ( (bc_ptr[BCVars::zvel_bc].hi(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].hi(1) == ERFBCType::ext_dir_ingested) );

    // Dirichlet on top or bottom plane
    bool zl_u_dir = ( (bc_ptr[BCVars::xvel_bc].lo(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].lo(2) == ERFBCType::ext_dir_ingested) );
    bool zh_u_dir = ( (bc_ptr[BCVars::xvel_bc].hi(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].hi(2) == ERFBCType::ext_dir_ingested) );

    bool zl_v_dir = ( (bc_ptr[BCVars::yvel_bc].lo(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].lo(2) == ERFBCType::ext_dir_ingested) );
    bool zh_v_dir = ( (bc_ptr[BCVars::yvel_bc].hi(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].hi(2) == ERFBCType::ext_dir_ingested) );


    // X-Dirichlet
    //***********************************************************************************
    if (xl_v_dir) {
        Box planexy = tbxxy; planexy.setBig(0, planexy.smallEnd(0) );
        tbxxy.growLo(0,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau12(i,j,k) = 0.5 * mu_eff * ( (u(i, j, k) - u(i, j-1, k))*dxInv[1] +
                                            (-(8./3.) * v(i-1,j,k) + 3. * v(i,j,k) - (1./3.) * v(i+1,j,k))*dxInv[0] );
        });
    }
    if (xh_v_dir) {
        Box planexy = tbxxy; planexy.setSmall(0, planexy.bigEnd(0) );
        tbxxy.growHi(0,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau12(i,j,k) = 0.5 * mu_eff * ( (u(i, j, k) - u(i, j-1, k))*dxInv[1] +
                                            -(-(8./3.) * v(i,j,k) + 3. * v(i-1,j,k) - (1./3.) * v(i-2,j,k))*dxInv[0] );
        });
    }

    if (xl_w_dir) {
        Box planexz = tbxxz; planexz.setBig(0, planexz.smallEnd(0) );
        tbxxz.growLo(0,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau13(i,j,k) = 0.5 * mu_eff * ( (u(i, j, k) - u(i, j, k-1))*dxInv[2] +
                                            (-(8./3.) * w(i-1,j,k) + 3. * w(i,j,k) - (1./3.) * w(i+1,j,k))*dxInv[0]);
        });
    }
    if (xh_w_dir) {
        Box planexz = tbxxz; planexz.setSmall(0, planexz.bigEnd(0) );
        tbxxz.growHi(0,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau13(i,j,k) = 0.5 * mu_eff * ( (u(i, j, k) - u(i, j, k-1))*dxInv[2] +
                                            -(-(8./3.) * w(i,j,k) + 3. * w(i-1,j,k) - (1./3.) * w(i-2,j,k))*dxInv[0]);
        });
    }

    // Y-Dirichlet
    //***********************************************************************************
    if (yl_u_dir) {
        Box planexy = tbxxy; planexy.setBig(1, planexy.smallEnd(1) );
        tbxxy.growLo(1,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau12(i,j,k) = 0.5 * mu_eff * ( (-(8./3.) * u(i,j-1,k) + 3. * u(i,j,k) - (1./3.) * u(i,j+1,k))*dxInv[1] +
                                            (v(i, j, k) - v(i-1, j, k))*dxInv[0] );
        });
    }
    if (yh_u_dir) {
        Box planexy = tbxxy; planexy.setSmall(1, planexy.bigEnd(1) );
        tbxxy.growHi(1,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau12(i,j,k) = 0.5 * mu_eff * ( -(-(8./3.) * u(i,j,k) + 3. * u(i,j-1,k) - (1./3.) * u(i,j-2,k))*dxInv[1] +
                                            (v(i, j, k) - v(i-1, j, k))*dxInv[0] );
        });
    }

    if (yl_w_dir) {
        Box planeyz = tbxyz; planeyz.setBig(1, planeyz.smallEnd(1) );
        tbxyz.growLo(1,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau23(i,j,k) = 0.5 * mu_eff * ( (v(i, j, k) - v(i, j, k-1))*dxInv[2] +
                                            (-(8./3.) * w(i,j-1,k) + 3. * w(i,j  ,k) - (1./3.) * w(i,j+1,k))*dxInv[1] );
        });
    }
    if (yh_w_dir) {
        Box planeyz = tbxyz; planeyz.setSmall(1, planeyz.bigEnd(1) );
        tbxyz.growHi(1,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau23(i,j,k) = 0.5 * mu_eff * ( (v(i, j, k) - v(i, j, k-1))*dxInv[2] +
                                            -(-(8./3.) * w(i,j  ,k) + 3. * w(i,j-1,k) - (1./3.) * w(i,j-2,k))*dxInv[1] );
        });
    }

    // Z-Dirichlet
    //***********************************************************************************
    if (zl_u_dir) {
        Box planexz = tbxxz; planexz.setBig(2, planexz.smallEnd(2) );
        tbxxz.growLo(2,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau13(i,j,k) = 0.5 * mu_eff * ( (-(8./3.) * u(i,j,k-1) + 3. * u(i,j,k) - (1./3.) * u(i,j,k+1))*dxInv[2] +
                                            (w(i, j, k) - w(i-1, j, k))*dxInv[0] );
        });
    }
    if (zh_u_dir) {
        Box planexz = tbxxz; planexz.setSmall(2, planexz.bigEnd(2) );
        tbxxz.growHi(2,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau13(i,j,k) = 0.5 * mu_eff * ( -(-(8./3.) * u(i,j,k) + 3. * u(i,j,k-1) - (1./3.) * u(i,j,k-2))*dxInv[2] +
                                            (w(i, j, k) - w(i-1, j, k))*dxInv[0] );
        });
    }

    if (zl_v_dir) {
        Box planeyz = tbxyz; planeyz.setBig(2, planeyz.smallEnd(2) );
        tbxyz.growLo(2,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau23(i,j,k) = 0.5 * mu_eff * ( (-(8./3.) * v(i,j,k-1) + 3. * v(i,j,k  ) - (1./3.) * v(i,j,k+1))*dxInv[2] +
                                            (w(i, j, k) - w(i, j-1, k))*dxInv[1] );
        });
    }
    if (zh_v_dir) {
        Box planeyz = tbxyz; planeyz.setSmall(2, planeyz.bigEnd(2) );
        tbxyz.growHi(2,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau23(i,j,k) = 0.5 * mu_eff * ( -(-(8./3.) * v(i,j,k  ) + 3. * v(i,j,k-1) - (1./3.) * v(i,j,k-2))*dxInv[2] +
                                            (w(i, j, k) - w(i, j-1, k))*dxInv[1] );
        });
    }

    // Fill the remaining cells
    //***********************************************************************************
    // Cell centered strains
    amrex::ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        tau11(i,j,k) = mu_eff * ( (u(i+1, j  , k  ) - u(i, j, k))*dxInv[0] - OneThird*er_arr(i,j,k) );
        tau22(i,j,k) = mu_eff * ( (v(i  , j+1, k  ) - v(i, j, k))*dxInv[1] - OneThird*er_arr(i,j,k) );
        tau33(i,j,k) = mu_eff * ( (w(i  , j  , k+1) - w(i, j, k))*dxInv[2] - OneThird*er_arr(i,j,k) );
    });

    // Off-diagonal strains
    amrex::ParallelFor(tbxxy,tbxxz,tbxyz,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        tau12(i,j,k) = 0.5 * mu_eff * ( (u(i, j, k) - u(i, j-1, k))*dxInv[1] + (v(i, j, k) - v(i-1, j, k))*dxInv[0] );
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        tau13(i,j,k) = 0.5 * mu_eff * ( (u(i, j, k) - u(i, j, k-1))*dxInv[2] + (w(i, j, k) - w(i-1, j, k))*dxInv[0] );
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        tau23(i,j,k) = 0.5 * mu_eff * ( (v(i, j, k) - v(i, j, k-1))*dxInv[2] + (w(i, j, k) - w(i, j-1, k))*dxInv[1] );
    });

}


void
ComputeStressVarVisc_N(Box& bxcc, Box& tbxxy, Box& tbxxz, Box& tbxyz, Real mu_eff,
                       const Array4<const Real>& K_turb,
                       const Array4<const Real>& u, const Array4<const Real>& v, const Array4<const Real>& w,
                       Array4<Real>& tau11, Array4<Real>& tau22, Array4<Real>& tau33,
                       Array4<Real>& tau12, Array4<Real>& tau13, Array4<Real>& tau23,
                       const Array4<const Real>& er_arr,
                       const BCRec* bc_ptr, const GpuArray<Real, AMREX_SPACEDIM>& dxInv)
{
    Real OneThird   = (1./3.);

    // Dirichlet on left or right plane
    bool xl_v_dir = ( (bc_ptr[BCVars::yvel_bc].lo(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].lo(0) == ERFBCType::ext_dir_ingested) );
    bool xh_v_dir = ( (bc_ptr[BCVars::yvel_bc].hi(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].hi(0) == ERFBCType::ext_dir_ingested) );

    bool xl_w_dir = ( (bc_ptr[BCVars::zvel_bc].lo(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].lo(0) == ERFBCType::ext_dir_ingested) );
    bool xh_w_dir = ( (bc_ptr[BCVars::zvel_bc].hi(0) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].hi(0) == ERFBCType::ext_dir_ingested) );

    // Dirichlet on front or back plane
    bool yl_u_dir = ( (bc_ptr[BCVars::xvel_bc].lo(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].lo(1) == ERFBCType::ext_dir_ingested) );
    bool yh_u_dir = ( (bc_ptr[BCVars::xvel_bc].hi(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].hi(1) == ERFBCType::ext_dir_ingested) );

    bool yl_w_dir = ( (bc_ptr[BCVars::zvel_bc].lo(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].lo(1) == ERFBCType::ext_dir_ingested) );
    bool yh_w_dir = ( (bc_ptr[BCVars::zvel_bc].hi(1) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::zvel_bc].hi(1) == ERFBCType::ext_dir_ingested) );

    // Dirichlet on top or bottom plane
    bool zl_u_dir = ( (bc_ptr[BCVars::xvel_bc].lo(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].lo(2) == ERFBCType::ext_dir_ingested) );
    bool zh_u_dir = ( (bc_ptr[BCVars::xvel_bc].hi(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::xvel_bc].hi(2) == ERFBCType::ext_dir_ingested) );

    bool zl_v_dir = ( (bc_ptr[BCVars::yvel_bc].lo(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].lo(2) == ERFBCType::ext_dir_ingested) );
    bool zh_v_dir = ( (bc_ptr[BCVars::yvel_bc].hi(2) == ERFBCType::ext_dir)          ||
                      (bc_ptr[BCVars::yvel_bc].hi(2) == ERFBCType::ext_dir_ingested) );


    // X-Dirichlet
    //***********************************************************************************
    if (xl_v_dir) {
        Box planexy = tbxxy; planexy.setBig(0, planexy.smallEnd(0) );
        tbxxy.growLo(0,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_12 = mu_eff + 0.25*( K_turb(i-1, j  , k, EddyDiff::Mom_h) + K_turb(i, j  , k, EddyDiff::Mom_h)
                                       + K_turb(i-1, j-1, k, EddyDiff::Mom_h) + K_turb(i, j-1, k, EddyDiff::Mom_h) );
            tau12(i,j,k) = 0.5 * mu_12 * ( (u(i, j, k) - u(i, j-1, k))*dxInv[1] +
                                           (-(8./3.) * v(i-1,j,k) + 3. * v(i,j,k) - (1./3.) * v(i+1,j,k))*dxInv[0] );
        });
    }
    if (xh_v_dir) {
        Box planexy = tbxxy; planexy.setSmall(0, planexy.bigEnd(0) );
        tbxxy.growHi(0,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_12 = mu_eff + 0.25*( K_turb(i-1, j  , k, EddyDiff::Mom_h) + K_turb(i, j  , k, EddyDiff::Mom_h)
                                       + K_turb(i-1, j-1, k, EddyDiff::Mom_h) + K_turb(i, j-1, k, EddyDiff::Mom_h) );
            tau12(i,j,k) = 0.5 * mu_12 * ( (u(i, j, k) - u(i, j-1, k))*dxInv[1] +
                                           -(-(8./3.) * v(i,j,k) + 3. * v(i-1,j,k) - (1./3.) * v(i-2,j,k))*dxInv[0] );
        });
    }

    if (xl_w_dir) {
        Box planexz = tbxxz; planexz.setBig(0, planexz.smallEnd(0) );
        tbxxz.growLo(0,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_13 = mu_eff + 0.25*( K_turb(i-1, j, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i-1, j, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau13(i,j,k) = 0.5 * mu_13 * ( (u(i, j, k) - u(i, j, k-1))*dxInv[2] +
                                           (-(8./3.) * w(i-1,j,k) + 3. * w(i,j,k) - (1./3.) * w(i+1,j,k))*dxInv[0]);
        });
    }
    if (xh_w_dir) {
        Box planexz = tbxxz; planexz.setSmall(0, planexz.bigEnd(0) );
        tbxxz.growHi(0,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_13 = mu_eff + 0.25*( K_turb(i-1, j, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i-1, j, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau13(i,j,k) = 0.5 * mu_13 * ( (u(i, j, k) - u(i, j, k-1))*dxInv[2] +
                                           -(-(8./3.) * w(i,j,k) + 3. * w(i-1,j,k) - (1./3.) * w(i-2,j,k))*dxInv[0]);
        });
    }

    // Y-Dirichlet
    //***********************************************************************************
    if (yl_u_dir) {
        Box planexy = tbxxy; planexy.setBig(1, planexy.smallEnd(1) );
        tbxxy.growLo(1,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_12 = mu_eff + 0.25*( K_turb(i-1, j  , k, EddyDiff::Mom_h) + K_turb(i, j  , k, EddyDiff::Mom_h)
                                       + K_turb(i-1, j-1, k, EddyDiff::Mom_h) + K_turb(i, j-1, k, EddyDiff::Mom_h) );
            tau12(i,j,k) = 0.5 * mu_12 * ( (-(8./3.) * u(i,j-1,k) + 3. * u(i,j,k) - (1./3.) * u(i,j+1,k))*dxInv[1] +
                                           (v(i, j, k) - v(i-1, j, k))*dxInv[0] );
        });
    }
    if (yh_u_dir) {
        Box planexy = tbxxy; planexy.setSmall(1, planexy.bigEnd(1) );
        tbxxy.growHi(1,-1);
        amrex::ParallelFor(planexy,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_12 = mu_eff + 0.25*( K_turb(i-1, j  , k, EddyDiff::Mom_h) + K_turb(i, j  , k, EddyDiff::Mom_h)
                                       + K_turb(i-1, j-1, k, EddyDiff::Mom_h) + K_turb(i, j-1, k, EddyDiff::Mom_h) );
            tau12(i,j,k) = 0.5 * mu_12 * ( -(-(8./3.) * u(i,j,k) + 3. * u(i,j-1,k) - (1./3.) * u(i,j-2,k))*dxInv[1] +
                                           (v(i, j, k) - v(i-1, j, k))*dxInv[0] );
        });
    }

    if (yl_w_dir) {
        Box planeyz = tbxyz; planeyz.setBig(1, planeyz.smallEnd(1) );
        tbxyz.growLo(1,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_23 = mu_eff + 0.25*( K_turb(i, j-1, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i, j-1, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau23(i,j,k) = 0.5 * mu_23 * ( (v(i, j, k) - v(i, j, k-1))*dxInv[2] +
                                           (-(8./3.) * w(i,j-1,k) + 3. * w(i,j  ,k) - (1./3.) * w(i,j+1,k))*dxInv[1] );
        });
    }
    if (yh_w_dir) {
        Box planeyz = tbxyz; planeyz.setSmall(1, planeyz.bigEnd(1) );
        tbxyz.growHi(1,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_23 = mu_eff + 0.25*( K_turb(i, j-1, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i, j-1, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau23(i,j,k) = 0.5 * mu_23 * ( (v(i, j, k) - v(i, j, k-1))*dxInv[2] +
                                           -(-(8./3.) * w(i,j  ,k) + 3. * w(i,j-1,k) - (1./3.) * w(i,j-2,k))*dxInv[1] );
        });
    }

    // Z-Dirichlet
    //***********************************************************************************
    if (zl_u_dir) {
        Box planexz = tbxxz; planexz.setBig(2, planexz.smallEnd(2) );
        tbxxz.growLo(2,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_13 = mu_eff + 0.25*( K_turb(i-1, j, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i-1, j, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau13(i,j,k) = 0.5 * mu_13 * ( (-(8./3.) * u(i,j,k-1) + 3. * u(i,j,k) - (1./3.) * u(i,j,k+1))*dxInv[2] +
                                           (w(i, j, k) - w(i-1, j, k))*dxInv[0] );
        });
    }
    if (zh_u_dir) {
        Box planexz = tbxxz; planexz.setSmall(2, planexz.bigEnd(2) );
        tbxxz.growHi(2,-1);
        amrex::ParallelFor(planexz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_13 = mu_eff + 0.25*( K_turb(i-1, j, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i-1, j, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau13(i,j,k) = 0.5 * mu_13 * ( -(-(8./3.) * u(i,j,k) + 3. * u(i,j,k-1) - (1./3.) * u(i,j,k-2))*dxInv[2] +
                                           (w(i, j, k) - w(i-1, j, k))*dxInv[0] );
        });
    }

    if (zl_v_dir) {
        Box planeyz = tbxyz; planeyz.setBig(2, planeyz.smallEnd(2) );
        tbxyz.growLo(2,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_23 = mu_eff + 0.25*( K_turb(i, j-1, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i, j-1, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau23(i,j,k) = 0.5 * mu_23 * ( (-(8./3.) * v(i,j,k-1) + 3. * v(i,j,k  ) - (1./3.) * v(i,j,k+1))*dxInv[2] +
                                           (w(i, j, k) - w(i, j-1, k))*dxInv[1] );
        });
    }
    if (zh_v_dir) {
        Box planeyz = tbxyz; planeyz.setSmall(2, planeyz.bigEnd(2) );
        tbxyz.growHi(2,-1);
        amrex::ParallelFor(planeyz,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_23 = mu_eff + 0.25*( K_turb(i, j-1, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                       + K_turb(i, j-1, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
            tau23(i,j,k) = 0.5 * mu_23 * ( -(-(8./3.) * v(i,j,k  ) + 3. * v(i,j,k-1) - (1./3.) * v(i,j,k-2))*dxInv[2] +
                                           (w(i, j, k) - w(i, j-1, k))*dxInv[1] );
        });
    }

    // Fill the remaining cells
    //***********************************************************************************
    // Cell centered strains
    amrex::ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        Real mu_11 = mu_eff + K_turb(i, j, k, EddyDiff::Mom_h);
        Real mu_22 = mu_11;
        Real mu_33 = mu_eff + K_turb(i, j, k, EddyDiff::Mom_v);
        tau11(i,j,k) = mu_11 * ( (u(i+1, j  , k  ) - u(i, j, k))*dxInv[0] - OneThird*er_arr(i,j,k) );
        tau22(i,j,k) = mu_22 * ( (v(i  , j+1, k  ) - v(i, j, k))*dxInv[1] - OneThird*er_arr(i,j,k) );
        tau33(i,j,k) = mu_33 * ( (w(i  , j  , k+1) - w(i, j, k))*dxInv[2] - OneThird*er_arr(i,j,k) );
    });

    // Off-diagonal strains
    amrex::ParallelFor(tbxxy,tbxxz,tbxyz,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        Real mu_12 = mu_eff + 0.25*( K_turb(i-1, j  , k, EddyDiff::Mom_h) + K_turb(i, j  , k, EddyDiff::Mom_h)
                                   + K_turb(i-1, j-1, k, EddyDiff::Mom_h) + K_turb(i, j-1, k, EddyDiff::Mom_h) );
        tau12(i,j,k) = 0.5 * mu_12 * ( (u(i, j, k) - u(i, j-1, k))*dxInv[1] + (v(i, j, k) - v(i-1, j, k))*dxInv[0] );
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        Real mu_13 = mu_eff + 0.25*( K_turb(i-1, j, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                   + K_turb(i-1, j, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
        tau13(i,j,k) = 0.5 * mu_13 * ( (u(i, j, k) - u(i, j, k-1))*dxInv[2] + (w(i, j, k) - w(i-1, j, k))*dxInv[0] );
    },
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
        Real mu_23 = mu_eff + 0.25*( K_turb(i, j-1, k  , EddyDiff::Mom_v) + K_turb(i, j, k  , EddyDiff::Mom_v)
                                   + K_turb(i, j-1, k-1, EddyDiff::Mom_v) + K_turb(i, j, k-1, EddyDiff::Mom_v) );
        tau23(i,j,k) = 0.5 * mu_23 * ( (v(i, j, k) - v(i, j, k-1))*dxInv[2] + (w(i, j, k) - w(i, j-1, k))*dxInv[1] );
    });


}
