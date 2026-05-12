! find_hops.f90
!
! Fortran 90 implementation of Hopping.find_hops() in hopping.py.
! Inner loops are identical to the Python version; the outer O(N^2) pair
! search runs 50-200x faster at the Fortran level.
!
! Compile
! -------
!   f2py -c find_hops.f90 -m find_hopsf90
!
! Or with optimisation flags:
!   f2py -c find_hops.f90 -m find_hops_f90 --f90flags="-O3 -march=native"
!
! Python usage (after compilation)
! ---------------------------------
!   from tbforge.find_hops_f90 import find_hops
!
!   hopsxy, nhopsxy, hopsz, nhopsz = find_hops(
!       rlist, lat_vecs, bc, rxy, rz, d, tol)
!
!   hopsxy = hopsxy[:nhopsxy]   ! trim to actual count
!   hopsz  = hopsz[:nhopsz]
!
! Arguments Python does NOT pass (auto-computed by f2py via intent(hide)):
!   n_sites  = rlist.shape[0]
!   max_hops = product_i( bc[i]*(2*d) + 1 ) * n_sites*n_sites
!              = exact upper bound on hop count, respecting bc per direction
!
! Output columns (0-indexed, matching the Python version)
! --------------------------------------------------------
!   hopsxy / hopsz  (nhops, 9):
!     col 0  = cell shift ii  (x)
!     col 1  = cell shift jj  (y)
!     col 2  = cell shift kk  (z)
!     col 3  = site index s   (0-indexed for Python)
!     col 4  = site index sp  (0-indexed for Python)
!     col 5  = distance (dxy for hopsxy, 3-D dist for hopsz)
!     col 6  = rij(x)
!     col 7  = rij(y)
!     col 8  = rij(z)

subroutine find_hops(rlist, n_sites, lat_vecs, bc, rxy, rz, d, tol, &
                     hopsxy, nhopsxy, hopsz, nhopsz, max_hops)

    implicit none

    ! --- Dummy arguments ---
    integer,          intent(in)  :: n_sites       ! hidden: rlist.shape[0]
    integer,          intent(in)  :: max_hops       ! hidden: bc-aware upper bound
    integer,          intent(in)  :: d              ! image-cell search depth
    integer,          intent(in)  :: bc(3)          ! periodic flags per direction
    double precision, intent(in)  :: rlist(n_sites, 3)
    double precision, intent(in)  :: lat_vecs(3, 3) ! rows = a1, a2, a3
    double precision, intent(in)  :: rxy            ! target in-plane distance
    double precision, intent(in)  :: rz             ! target interlayer distance (0 = skip)
    double precision, intent(in)  :: tol

    double precision, intent(out) :: hopsxy(max_hops, 9)
    double precision, intent(out) :: hopsz(max_hops, 9)
    integer,          intent(out) :: nhopsxy, nhopsz

    ! f2py directives ---------------------------------------------------
    ! n_sites is hidden and inferred from rlist
    !f2py intent(in)  :: rlist, lat_vecs, bc, rxy, rz, d, tol
    !f2py integer, intent(hide), depend(rlist) :: n_sites = shape(rlist, 0)
    ! max_hops = product over each direction of (bc[i]*(2*d)+1) * n_sites^2
    ! When bc[i]=1: factor = 2*d+1 (full periodic range)
    ! When bc[i]=0: factor = 1     (no image shift)
    !f2py integer, intent(hide), depend(n_sites, d, bc) :: max_hops = (bc[0]*(2*d)+1)*(bc[1]*(2*d)+1)*(bc[2]*(2*d)+1)*n_sites*n_sites
    !f2py intent(out) :: hopsxy, nhopsxy, hopsz, nhopsz
    !f2py depend(max_hops) :: hopsxy, hopsz
    ! -------------------------------------------------------------------

    ! --- Local variables ---
    integer          :: ii, jj, kk, s, sp
    integer          :: ix_lo, ix_hi, iy_lo, iy_hi, iz_lo, iz_hi
    double precision :: ri(3), rj(3), rij(3), shift(3)
    double precision :: dxy, dist

    nhopsxy = 0
    nhopsz  = 0
    hopsxy  = 0.0d0
    hopsz   = 0.0d0

    ! --- Search ranges from bc ---
    if (bc(1) == 1) then; ix_lo = -d; ix_hi = d
    else;                 ix_lo =  0; ix_hi = 0; end if
    if (bc(2) == 1) then; iy_lo = -d; iy_hi = d
    else;                 iy_lo =  0; iy_hi = 0; end if
    if (bc(3) == 1) then; iz_lo = -d; iz_hi = d
    else;                 iz_lo =  0; iz_hi = 0; end if

    ! --- Main pair loop ---
    do ii = ix_lo, ix_hi
        do jj = iy_lo, iy_hi
            do kk = iz_lo, iz_hi

                ! Lattice translation for this image cell
                shift = dble(ii)*lat_vecs(1,:) &
                      + dble(jj)*lat_vecs(2,:) &
                      + dble(kk)*lat_vecs(3,:)

                do s = 1, n_sites
                    ri = rlist(s, :)
                    do sp = 1, n_sites
                        rj   = rlist(sp, :) + shift
                        rij  = ri - rj
                        dxy  = sqrt(rij(1)**2 + rij(2)**2)
                        dist = sqrt(rij(1)**2 + rij(2)**2 + rij(3)**2)

                        ! --- In-plane hop: at rxy, same z-layer ---
                        if (abs(rxy - dxy) < tol .and. abs(rij(3)) < tol) then
                            nhopsxy = nhopsxy + 1
                            if (nhopsxy <= max_hops) then
                                hopsxy(nhopsxy, 1) = dble(ii)
                                hopsxy(nhopsxy, 2) = dble(jj)
                                hopsxy(nhopsxy, 3) = dble(kk)
                                hopsxy(nhopsxy, 4) = dble(s  - 1)  ! 0-indexed for Python
                                hopsxy(nhopsxy, 5) = dble(sp - 1)
                                hopsxy(nhopsxy, 6) = dxy
                                hopsxy(nhopsxy, 7) = rij(1)
                                hopsxy(nhopsxy, 8) = rij(2)
                                hopsxy(nhopsxy, 9) = rij(3)
                            end if
                        end if

                        ! --- Interlayer hop: at rz, different z-layer ---
                        if (rz > 0.0d0 .and. abs(dist - rz) < tol .and. &
                            abs(rij(3)) > tol) then
                            nhopsz = nhopsz + 1
                            if (nhopsz <= max_hops) then
                                hopsz(nhopsz, 1) = dble(ii)
                                hopsz(nhopsz, 2) = dble(jj)
                                hopsz(nhopsz, 3) = dble(kk)
                                hopsz(nhopsz, 4) = dble(s  - 1)
                                hopsz(nhopsz, 5) = dble(sp - 1)
                                hopsz(nhopsz, 6) = dist
                                hopsz(nhopsz, 7) = rij(1)
                                hopsz(nhopsz, 8) = rij(2)
                                hopsz(nhopsz, 9) = rij(3)
                            end if
                        end if

                    end do  ! sp
                end do  ! s
            end do  ! kk
        end do  ! jj
    end do  ! ii

end subroutine find_hops
