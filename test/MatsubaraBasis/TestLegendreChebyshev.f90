! program TestLegendreChebyshev
!   use Common
!   implicit none
!   integer :: ic,jc,nc
!   complex*16 :: umat(0:9,0:9), umatinv(0:9, 0:9)

!   call Legendre2Chebyshev(10, umat)

!   call dcmplx_matinv(umat, umatinv, 10, 10)

!   nc=10
!   open(unit=8, file='trans.dat')
!   do ic=nc-1, 0, -1
!     write(8, '(10(f20.12, 2x))') (dble(umatinv(ic, jc)), jc=nc-1, 0, -1)
!   enddo
!   close(8)
! end program TestLegendreChebyshev



program TestLegendreChebyshev
  use MatsubaraBasis
  implicit none
  integer :: ic,jc,nc
  complex*16 :: vecin(0:9), vecout(0:9)
  complex*16 :: umat(0:9,0:9), umatinv(0:9, 0:9)

  nc=10
  do ic=0, nc-1
    vecin=0.0d0
    vecin(ic)=1.0d0
    call LegendreC2ChebyshevC(10, vecin, vecout)
    print '(10(f20.12, 2x))', dble(vecout)
  enddo

  print *, ''
  print *, ''
  


  do ic=0, nc-1
    vecin=0.0d0
    vecin(ic)=1.0d0
    call ChebyshevC2LegendreC(10, vecin, vecout)
    print '(10(f20.12, 2x))', dble(vecout)
  enddo

  
end program TestLegendreChebyshev


