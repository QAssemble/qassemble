program TestMoment
  implicit none

  integer :: inu, itau,ii,ierr, l,itheta
  double precision :: beta, pi,moment(3),erg,xx

  integer*8 :: ntau,nnu
  complex*16 :: ai

  integer*8, allocatable :: null
  double precision, allocatable :: tau(:),taurad(:),nu(:), theta(:), thetain(:)
  complex*16,allocatable :: fnu(:),fnuout(:),m1nu(:),m2nu(:),m3nu(:),ftau(:),m1tau(:),m2tau(:),m3tau(:),ftau_tot(:), ftau_tot_w_weight(:), ftau_tot_analytic(:), ftau_exa(:), chebyshev_temp(:),chebyshev_coeff(:), chebyshev_coeff2(:), ftheta(:),chebyshev_array(:), ftau_temp(:)


  integer, external :: ttind
  double precision, external :: eulerpolynomial, factorial_int, BernoulliPolynomial

  nnu=3000
  ntau=30000
  erg=2.0d0

  ai=dcmplx(0.0d0, 1.0d0)

  allocate(tau(0:(ntau-1))) 
  tau=0.0d0
  
  allocate(theta((-ntau):(ntau-1))) ! *2 for the chebyshev polynomial coefficient evaluation
  theta=0.0d0
  allocate(taurad(0:(ntau-1)))
  taurad=0.0d0

  allocate(ftheta((-ntau):(ntau-1)))
  ftheta=0.0d0    

  allocate(ftau(0:(ntau-1)))
  ftau=0.0d0

  allocate(chebyshev_temp((-ntau):(ntau-1)))
  chebyshev_temp=0.0d0

  allocate(chebyshev_array((-ntau+1):(ntau-1)))
  chebyshev_array=0.0d0  

  allocate(chebyshev_coeff(0:(ntau-1)))
  chebyshev_coeff=0.0d0    

  allocate(chebyshev_coeff2(0:(ntau-1)))
  chebyshev_coeff2=0.0d0    

  allocate(ftau_tot(0:(ntau-1)))
  ftau_tot=0.0d0
  allocate(ftau_tot_w_weight(0:(ntau-1)))
  ftau_tot_w_weight=0.0d0

  allocate(ftau_tot_analytic(0:(ntau-1)))
  ftau_tot_analytic=0.0d0    

  allocate(ftau_exa(0:(ntau-1)))
  ftau_exa=0.0d0    
  allocate(m1tau(0:(ntau-1)))
  m1tau=0.0d0
  allocate(m2tau(0:(ntau-1)))
  m2tau=0.0d0
  allocate(m3tau(0:(ntau-1)))
  m3tau=0.0d0      

  allocate(nu((-nnu+1):(nnu-1)))
  nu=0.0d0

  allocate(fnu((-nnu+1):(nnu-1)))
  fnu=0.0d0
  allocate(fnuout((-nnu+1):(nnu-1)))
  fnuout=0.0d0  
  allocate(m1nu((-nnu+1):(nnu-1)))
  m1nu=0.0d0
  allocate(m2nu((-nnu+1):(nnu-1)))
  m2nu=0.0d0
  allocate(m3nu((-nnu+1):(nnu-1)))
  m3nu=0.0d0  

  beta=1.0d0/(8.617333262145d-5*300.0d0)
  pi=datan2(1.0d0,1.0d0)*4.0d0

  do inu=-nnu+1, nnu-1
    nu(inu)=2.0d0*pi/beta*inu
  enddo

  do inu=-nnu+1, nnu-1
    fnu(inu)=1.0d0/(nu(inu)*ai-erg)
    if (inu .ne. 0) then
      m1nu(inu)=1.0d0/(nu(inu)*ai)
      m2nu(inu)=1.0d0/(nu(inu)*ai)**2
      m3nu(inu)=1.0d0/(nu(inu)*ai)**3
    endif
  enddo

  do itheta=0, ntau-1
    theta(itheta)=pi*(itheta+0.5d0)/dble(ntau)
    theta(-itheta-1)=-theta(itheta)    
  enddo

  do itau=0, ntau-1
    itheta=ttind(itau, ntau)
    tau(itau)=beta/2.0d0*(dcos(theta(itheta))+1)
    taurad(itau)=pi*(dcos(theta(itheta))+1)
  enddo

  open(unit=8, file='tau.dat')
  do itau=0, ntau-1  
    write(8,*) itau, tau(itau), taurad(itau)
  enddo
  close(8)

  open(unit=8, file='theta.dat')
  do itheta=-ntau, ntau-1  
    write(8,*) itheta, theta(itheta)
  enddo
  close(8)  

  call finufft1d2(ntau,taurad,ftau,-1, 1.0d-12, 2*nnu-1,fnu,null,ierr)
  call finufft1d2(ntau,taurad,m1tau,-1, 1.0d-12, 2*nnu-1,m1nu,null,ierr)
  call finufft1d2(ntau,taurad,m2tau,-1, 1.0d-12, 2*nnu-1,m2nu,null,ierr)
  call finufft1d2(ntau,taurad,m3tau,-1, 1.0d-12, 2*nnu-1,m3nu,null,ierr)

  moment(1)=1.0d0
  moment(2) &
    =moment(2) &
    +(fnu(-nnu+1)+fnu(nnu-1)) &
    /2.0d0*(nu(nnu-1)*ai)**2

  moment(3) &
    =moment(3) &
    +(-fnu(-nnu+1)+fnu(nnu-1)-1.0d0*2.0d0/(nu(nnu-1)*ai))/2.0d0*(nu(nnu-1)*ai)**3
  print *, moment  

  do itau=0, ntau-1
    ftau_tot(itau)=ftau(itau)/beta-moment(1)*m1tau(itau)/beta-moment(2)*m2tau(itau)/beta-moment(3)*m3tau(itau)/beta
    do ii=1, 3
      xx=tau(itau)/beta
      ftau_tot(itau)=ftau_tot(itau)+moment(ii)*(beta)**(ii-1)*(-1)**(ii-1)/factorial_int(ii)*BernoulliPolynomial(xx, ii)      
    enddo
    ftau_tot_w_weight(itau)=ftau_tot(itau)*dsqrt(tau(itau)*(beta-tau(itau)))*pi/ntau
  enddo

  call ftau_nonint(ntau, tau, beta, erg, ftau_exa)  

  ! open(unit=8, file='ftau.dat')
  ! do itau=0, ntau-1
  !   write(8,'(i6, 7(f20.12, 3x))') itau, ftau_tot(itau), ftau_exa(itau)
  ! enddo
  ! close(8)
  
  do itheta=0, ntau-1
    itau=ttind(itheta, ntau)
    ftheta(itheta)=ftau_tot(itau)
    ftheta(-itheta-1)=ftheta(itheta)    
  enddo

  call finufft1d1(2*ntau,theta,ftheta,1, 1.0d-12, 2*ntau,chebyshev_temp,null,ierr)

  chebyshev_coeff=chebyshev_temp(0:(ntau-1))/ntau
  chebyshev_coeff(0)=chebyshev_coeff(0)/2.0d0

  do l=0,  ntau-1
    do itau=0, ntau-1    
      if (l .eq. 0) then
        chebyshev_coeff2(l)=chebyshev_coeff2(l)+1.0d0/dble(ntau)*ftau_tot(itau)*dcos(l*dacos(2.0*tau(itau)/beta-1.0d0))
      else
        chebyshev_coeff2(l)=chebyshev_coeff2(l)+2.0d0/dble(ntau)*ftau_tot(itau)*dcos(l*dacos(2.0*tau(itau)/beta-1.0d0))
      endif
    enddo
  enddo
  
  open(unit=8, file='chebyshev.dat')
  do l=0, ntau-1
    write(8,'(i5, 3(2f20.12, 2x))') l, chebyshev_coeff(l), chebyshev_coeff2(l), chebyshev_coeff(l)-chebyshev_coeff2(l)
  enddo
  close(8)
  

  allocate(thetain(0:(ntau-1)))
  allocate(ftau_temp(0:(ntau-1)))  

  thetain=theta(0:(ntau-1))
  ! thetain=(/0.0d0, beta/2.0d0, beta/)-1.0d0)  

  do itau=0, ntau-1
    thetain(itau)=dacos(2.0*tau(itau)/beta-1.0d0)
  enddo

  do itau=-ntau+1, ntau-1
    if (itau .eq. 0) then
      chebyshev_array(itau)=chebyshev_coeff(itau)
    elseif (itau .gt. 0) then
      chebyshev_array(itau)=chebyshev_coeff(itau)/2.0d0
    else
      chebyshev_array(itau)=chebyshev_coeff(-itau)/2.0d0
    endif
  enddo
  

  call finufft1d2(ntau,thetain,ftau_temp,-1, 1.0d-12, 2*ntau-1,chebyshev_array,null,ierr)


  do itau=0, ntau-1
    do l=0,  ntau-1    
      ftau_tot_analytic(itau)=ftau_tot_analytic(itau)+chebyshev_coeff(l)*dcos(l*dacos(2.0*tau(itau)/beta-1.0d0))
    enddo
  enddo
  

! ! ftau_temp=0.0d0
! ! do l=0,  ntau-1    
! !   ftau_temp(1)=ftau_temp(1)+chebyshev_coeff(l)
! !   ftau_temp(2)=ftau_temp(2)+chebyshev_coeff(l)*dcos(l*pi/2.0d0)
! ! enddo

! ! print *, ftau_temp

!   call ftau_nonint(ntau, tau, beta, erg, ftau_exa)  

  open(unit=8, file='ftau.dat')
  do itau=0, ntau-1
    write(8,'(i6, 9(f20.12, 3x))') itau, ftau_tot(itau), ftau_exa(itau), ftau_tot_analytic(itau), ftau_temp(itau)
  enddo
  close(8)


!   call finufft1d1(ntau,taurad,ftau_tot_w_weight,1, 1.0d-12, 2*nnu-1,fnuout,null,ierr)

!   open(unit=8, file='fnu.dat')  
!   do inu=0, nnu-1
!     write(8,'(i6, 5(f12.6, 3x))') inu, fnu(inu), fnuout(inu)
!   enddo
!   close(8)
!   print *, cdabs(fnuout(1))/cdabs(fnu(1))



end program TestMoment


subroutine ftau_nonint(ntau, tau, beta, energy, ftau)
  implicit none


  integer, intent(in) :: ntau
  double precision, intent(in) :: tau(ntau), energy, beta


  integer :: unitnum
  double precision :: machep,taumod,taunew
  complex*16 :: ftau(ntau)


  integer :: itau


  machep = epsilon ( machep )

  do itau=1, ntau
    taumod=modulo(tau(itau), beta)
    unitnum=nint(tau(itau)-taumod)/beta
    if (taumod .lt. machep) then
      unitnum=unitnum-1
    endif
    taunew=tau(itau)-beta*unitnum          
    if (energy .gt. 0) then
      ftau(itau)=(-1)*dexp(-energy*taunew)*(1+1.0d0/(dexp(energy*beta)-1))
    else
      ftau(itau)=(-1)*dexp(-energy*(taunew-beta))*(1.0d0/(dexp(energy*beta)-1))
    endif
  enddo
end subroutine ftau_nonint



subroutine btau_nonint(ntau, tau, beta, energy, ftau)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency  
  implicit none


  integer, intent(in) :: ntau
  double precision, intent(in) :: tau(0:(ntau-1)), energy, beta


  integer :: unitnum
  double precision :: machep,taumod,taunew
  complex*16 :: ftau(ntau)


  integer :: itau

  do itau=0, ntau-1
    ftau(itau)=dexp(-energy*tau(itau))*(1-1.0d0/(dexp(energy*beta)-1))-dexp(energy*(tau(itau)-beta))*(1.0d0/(dexp(-energy*beta)-1))
  enddo
end subroutine btau_nonint






double precision function BernoulliPolynomial(x, n)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x
  double precision :: xmat(7), coeff(7)

  xmat=(/x**6, x**5, x**4, x**3, x**2, x, 1.0d0/)
  BernoulliPolynomial=0.0d0
  if (n .eq. 0) then
    BernoulliPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       0.0d0,       0.0d0,       0.0d0,       1.0d0/))
  elseif (n .eq. 1) then
    BernoulliPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       0.0d0,       0.0d0,       1.0d0,-1.0d0/2.0d0/))
  elseif (n .eq. 2) then
    BernoulliPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       0.0d0,       1.0d0,      -1.0d0, 1.0d0/6.0d0/))
  elseif (n .eq. 3) then
    BernoulliPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       1.0d0,-3.0d0/2.0d0, 1.0d0/2.0d0,       0.0d0/))
  elseif (n .eq. 4) then
    BernoulliPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       1.0d0,      -2.0d0,       1.0d0,       0.0d0, 1.0d0/3.0d1/))
  elseif (n .eq. 5) then
    BernoulliPolynomial=sum(xmat*(/      0.0d0,       1.0d0,-5.0d0/2.0d0, 5.0d0/3.0d0,       0.0d0,-1.0d0/6.0d0,       0.0d0/))
  elseif (n .eq. 6) then
    BernoulliPolynomial=sum(xmat*(/      1.0d0,      -3.0d0, 5.0d0/2.0d0,       0.0d0,-1.0d0/2.0d0,       0.0d0, 1.0d0/42.0d0/))
  endif

end function BernoulliPolynomial

Double precision function EulerPolynomial(x, n)
  implicit none
  integer, intent(in) :: n
  double precision, intent(in) :: x
  double precision :: xmat(7), coeff(7)

  EulerPolynomial=0.0d0

  xmat=(/x**6, x**5, x**4, x**3, x**2, x, 1.0d0/)
  if (n .eq. 0) then
    EulerPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       0.0d0,       0.0d0,       0.0d0,       1.0d0/))
  elseif (n .eq. 1) then
    EulerPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       0.0d0,       0.0d0,       1.0d0,-1.0d0/2.0d0/))
  elseif (n .eq. 2) then
    EulerPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       0.0d0,       1.0d0,      -1.0d0,       0.0d0/))
  elseif (n .eq. 3) then
    EulerPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       0.0d0,       1.0d0,-3.0d0/2.0d0,        0.d0, 1.0d0/4.0d0/))
  elseif (n .eq. 4) then
    EulerPolynomial=sum(xmat*(/      0.0d0,       0.0d0,       1.0d0,      -2.0d0,       0.0d0,       1.0d0,       0.0d0/))
  elseif (n .eq. 5) then
    EulerPolynomial=sum(xmat*(/      0.0d0,       1.0d0,-5.0d0/2.0d0,       0.0d0, 5.0d0/2.0d0,       0.0d0,-1.0d0/2.0d0/))
  elseif (n .eq. 6) then
    EulerPolynomial=sum(xmat*(/      1.0d0,      -3.0d0,       0.0d0,       5.0d0,       0.0d0,      -3.0d0,        0.0d0/))
  endif

end function EulerPolynomial



DOUBLE PRECISION function factorial_int(j)
  IMPLICIT NONE
  INTEGER, intent(in) :: j
  INTEGER :: i
  DOUBLE PRECISION :: x
  if (j<0) print *, &
    "factorial_int defined only for non-negative numbers!"
  x=1
  factorial_int = x
  if (j.eq.1) return
  DO i=2,j
    x = x*i
  END DO
  factorial_int = x
  return
end function factorial_int



integer function ttind(itheta, ntau)
! both possible    
! itau in, itheta out
! itheta in, itau out    
  implicit none
  integer, intent(in) :: itheta, ntau
  if (itheta .ge. 0) then
    ttind=ntau-1-itheta
  else
    ttind=-ntau-1-itheta
  endif

end function ttind


