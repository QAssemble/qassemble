subroutine single_ptl_green_moment(norb,ns,nk,nomega,ff, omega, moment)
  implicit none
  integer, intent(in) :: norb, nk,ns,nomega
  complex*16, intent(in) :: ff(norb,norb,ns,nk,0:(nomega-1)), omega(0:(nomega-1))

  complex*16,intent(out) :: moment(3,norb,norb,ns,nk)

  integer :: iorb, jorb, ik,is

  moment=0.0d0
  do ik=1, nk
    do is=1, ns
      do iorb=1, norb
        do jorb=1, norb
        
          if (iorb .eq. jorb) then
            moment(1,iorb,jorb,is,ik)=1.0d0
          else
            moment(1,iorb,jorb,is,ik)=0.0d0
          endif
          
          moment(2,iorb,jorb,is,ik) &
            =moment(2,iorb,jorb,is,ik) &
            +(gimp(iorb,jorb,is,ik,nomega-1)+dconjg(gimp(jorb,iorb,is,ik,nomega-1))) &
            /2.0d0*omega(n_omega-1)**2
          
          moment(3,iorb,jorb,is,ik) &
            =moment(3,iorb,jorb,is,ik) &
            +(gimp(iorb,jorb,is,ik,nomega-1)-dconjg(gimp(jorb,iorb,is,ik,nomega-1))-moment(1,iorb,jorb,is,ik)*2.0d0/omega(nomega-1))/2.0d0*omega(nomega-1)**3                            
        enddo
      enddo
    enddo
  enddo
  
end subroutine single_ptl_green_moment


subroutine boson_moment(norb,nk,nnu,ff, nu, moment)
  implicit none
  integer, intent(in) :: norb, nk,nnu
  complex*16, intent(in) :: ff(norb,norb,nk,0:(nnu-1)), nu(0:(nnu-1))
  complex*16,intent(out) :: moment(norb,norb,nk)

  moment=ff(:,:,:,nnu-1)*nu(n_nu-1)**2
  
end subroutine boson_moment


subroutine fermion_mat_omega_to_tau(norb,ns,nk,nomega,omega,fomega,moment,ntau,tau,ftau)
  implicit none

  integer, intent(in) :: norb,ns,nomega,ntau,nk
  double precision, intent(in) :: tau(0:ntau)
  complex*16, intent(in) :: omega(0:(nomega-1)),moment(3,norb,norb,ns,nk),fomega(norb,norb,ns,nk,0:(nomega-1))
  complex*16, intent(out) :: ftau(norb,norb,ns,nk,0:ntau)

  integer :: iorb, jorb,itau,iomega,ik,is
  double precision :: beta, pi


  pi=datan2(1.0d0,1.0d0)*4.0d0
  beta=pi/dimag(omega(1))
  print *, "ntau", ntau
  print *, "norb", norb
  print *, "nomega", nomega
  print *, "nk", nk  
  ! print *, omega(1), omega(nomega)
  ! print *, fomega(1,1,1), fomega(nomega,1,1)
  ! print *, tau(0)/beta, tau(ntau)/beta

  do iorb=1, norb
    do jorb=1, norb
      do is=1, ns
        do ik=1, nk
          do itau=0, ntau
            do iomega=0, nomega-1
! print *,iorb,jorb,itau
              ftau(iorb,jorb,is,ik,itau) &
                =ftau(iorb,jorb,is,ik,itau) &
                +1.0d0/beta*cdexp(-tau(itau)*omega(iomega)) &
                *( &
                fomega(iorb,jorb,is,ik,iomega) & 
                -moment(1,iorb,jorb,is,ik)/omega(iomega) &
                -moment(2,iorb,jorb,is,ik)/(omega(iomega))**2 &
                -moment(3,iorb,jorb,is,ik)/(omega(iomega))**3 &                
                ) &
                +1.0d0/beta*cdexp(tau(itau)*omega(iomega)) &
                *( &
                dconjg(fomega(jorb,iorb,is,ik,iomega)) &
                +dconjg(moment(1,jorb,iorb,is,ik))/omega(iomega) &
                -dconjg(moment(2,jorb,iorb,is,ik))/(omega(iomega))**2 &
                +dconjg(moment(3,jorb,iorb,is,ik))/(omega(iomega))**3 &                
                )
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
end subroutine fermion_mat_omega_to_tau


subroutine boson_mat_nu_to_tau(norb,nk,nnu,nu,fnu,moment,ntau,tau,ftau)

  implicit none
  integer, intent(in) :: norb,nnu,ntau,nk
  double precision, intent(in) :: tau(0:ntau)
  complex*16, intent(in) :: nu(0:nnu-1),moment(norb,norb,nk),fnu(norb,norb,nk,0:(nnu-1))
  complex*16, intent(out) :: ftau(norb,norb,nk,0:ntau)
  
  integer :: iorb, jorb,itau,inu,ik
  double precision :: beta, pi

  pi=datan2(1.0d0,1.0d0)*4.0d0
  beta=2.0d0*pi/dimag(nu(1))
  ftau=0.0d0

  do iorb=1,num_orb
    do iorb=1,num_orb
      do ik=1, nk
        do itau=0, n_tau
          do inu=0, nnu-1
            if (inu .eq. 0) then
              ftau(iorb,jorb,ik,itau) &
              =ftau(iorb,jorb,ik,itau)+1.0d0/beta*fnu(iorb1,iorb2,ik,inu)
            else
              ftau(iorb,jorb,ik,itau)=ftau(iorb,jorb,ik,itau) &
                +2.0d0/beta*dcos(dimag(nu(inu))*tau(itau))*( &
                fnu(iorb,jorb,ik,inu) &
                -moment(iorb,jorb,ik)/(nu(inu))**2 &
                )
            endif
          enddo
!     asymptotic
          ftau(iorb,jorb,ik,itau) &
            =ftau(iorb,jorb,ik,itau) &
            -moment(iorb,jorb,ik) &
            *beta/2.0d0 &
            *((tau(itau)/beta)**2-(tau(itau)/beta)+1.0d0/6.0d0)
        enddo
      enddo
    enddo
  enddo
end subroutine boson_mat_nu_to_tau



subroutine fermion_omega_to_tau(nomega,omega,fomega,moment,ntau,tau,ftau)
  implicit none

  integer, intent(in) :: nomega,ntau
  double precision, intent(in) :: tau(0:ntau),moment(3)
  complex*16, intent(in) :: omega(0:(nomega-1)),fomega(0:(nomega-1))
  complex*16, intent(out) :: ftau(0:ntau)

  integer :: itau,iomega
  double precision :: beta, pi


  pi=datan2(1.0d0,1.0d0)*4.0d0
  beta=pi/dimag(omega(1))
! print *, "ntau", ntau
! print *, "norb", norb
! print *, "nomega", nomega
! print *, omega(1), omega(nomega)
! print *, fomega(1,1,1), fomega(nomega,1,1)
  print *, tau(0)/beta, tau(ntau)/beta
  print *, moment
  print *, beta
  ftau=0.0d0

  do itau=0, ntau
    do iomega=0, nomega-1
      ftau(itau) &
        =ftau(itau) &
        +1.0d0/beta*cdexp(-tau(itau)*omega(iomega)) &
        *( &
        fomega(iomega) & 
        -moment(1)/omega(iomega) &
        -moment(2)/(omega(iomega))**2 &
        -moment(3)/(omega(iomega))**3 &                
        ) &
        +1.0d0/beta*cdexp(tau(itau)*omega(iomega)) &
        *( &
        dconjg(fomega(iomega)) &
        +moment(1)/omega(iomega) &
        -moment(2)/(omega(iomega))**2 &
        +moment(3)/(omega(iomega))**3 &                
        )
    enddo

    ftau(itau) &
      =ftau(itau) &
      -moment(1)/2.0d0 &
      +moment(2)*beta/2.0d0 &
      *(tau(itau)/beta-1.0d0/2.0d0) &
      -moment(3)*beta**2/4.0d0 &
      *((tau(itau)/beta)**2-(tau(itau)/beta)) 
  enddo
end subroutine fermion_omega_to_tau


subroutine boson_nu_to_tau(nnu,nu,fnu,moment,ntau,tau,ftau)
  implicit none

  integer, intent(in) :: nnu,ntau
  double precision, intent(in) :: tau(0:ntau),moment(1)
  complex*16, intent(in) :: nu(0:(nnu-1)),fnu(0:(nnu-1))
  complex*16, intent(out) :: ftau(0:ntau)

  integer :: itau,inu
  double precision :: beta, pi


  pi=datan2(1.0d0,1.0d0)*4.0d0
  beta=2.0d0*pi/dimag(nu(1))
! print *, "ntau", ntau
! print *, "norb", norb
! print *, "nnu", nnu
! print *, nu(1), nu(nnu)
! print *, fnu(1,1,1), fnu(nnu,1,1)
  print *, tau(0)/beta, tau(ntau)/beta
  print *, moment
  print *, beta
  print *, fnu(1:3)
  print *, nu(1:3)  
  ftau=0.0d0

  do itau=0, ntau
    do inu=0, nnu-1
      if (inu .eq. 0) then
        ftau(itau)=ftau(itau)+1.0d0/beta*fnu(inu)
      else
        ftau(itau) &
          =ftau(itau) &
          +2.0d0/beta*dcos(tau(itau)*dimag(nu(inu))) &
          *( &
          fnu(inu) & 
          -moment(1)/(nu(inu))**2 &
          )
      endif
    enddo
    ftau(itau) &
      =ftau(itau) &
      -moment(1)*beta/2.0d0 &
      *((tau(itau)/beta)**2-(tau(itau)/beta)+1.0d0/6.0d0)
  enddo
end subroutine boson_nu_to_tau



subroutine tau_to_freq(ntau,tau,ftau,nf,freq,ff)
  implicit none
  integer, intent(in) :: ntau,nf
  double precision, intent(in) :: tau(0:ntau)
  complex*16, intent(in) :: ftau(0:ntau), freq(0:(nf-1))
  complex*16, intent(out) :: ff(0:(nf-1))

  integer :: if,itau
  complex*16 :: temp(0:ntau),temp2(0:ntau)

  ff=0.0d0

  do if=0, nf-1
    temp=0.0d0
    temp2=0.0d0
    do itau=0, ntau
      temp(itau)=ftau(itau)*cdexp(freq(if)*tau(itau))
    enddo
    call fderiv_dcmplx(-1,ntau+1,tau(0),temp(0),temp2(0))
    ff(if)=temp2(ntau)
  enddo
end subroutine tau_to_freq



subroutine fermion_mat_tau_to_freq(norb,ns,nk,ntau,tau,ftau,nf,freq,ff)
  implicit none
  integer, intent(in) :: norb,ns,nk,ntau,nf
  double precision, intent(in) :: tau(0:ntau)
  complex*16, intent(in) :: ftau(norb,norb,ns,nk,0:ntau), freq(0:(nf-1))
  complex*16, intent(out) :: ff(norb,norb,ns,nk,0:(nf-1))

  integer :: iorb,jorb,itau,iff,is
  complex*16 :: tempt(0:ntau),tempf(0:(nf-1))

  ff=0.0d0

  do iorb=1, norb
    do jorb=1, norb
      do is=1, ns
        do ik=1, nk
          tempt=0.0d0
          do itau=0, ntau
            tempt(itau)=ftau(iorb,jorb,is,ik,itau)
          enddo
          
          call tau_to_freq(ntau,tau,tempt,nf,freq,tempf)
          
          do iff=0, nf-1
            ff(iorb,jorb,is,ik,iff)=tempf(iff)
          enddo
        enddo
      enddo
    enddo
  enddo
      
end subroutine fermion_mat_tau_to_freq


subroutine boson_mat_tau_to_freq(norb,nk,ntau,tau,ftau,nf,freq,ff)
  implicit none
  integer, intent(in) :: norb,nk,ntau,nf
  double precision, intent(in) :: tau(0:ntau)
  complex*16, intent(in) :: ftau(norb,norb,nk,0:ntau), freq(0:(nf-1))
  complex*16, intent(out) :: ff(norb,norb,nk,0:(nf-1))

  integer :: iorb,jorb,itau,iff
  complex*16 :: tempt(0:ntau),tempf(0:(nf-1))

  ff=0.0d0
  
  do iorb=1, norb
    do jorb=1, norb
      do ik=1, nk
        tempt=0.0d0
        do itau=0, ntau
          tempt(itau)=ftau(iorb,jorb,ik,itau)
        enddo
        
        call tau_to_freq(ntau,tau,tempt,nf,freq,tempf)
        
        do iff=0, nf-1
          ff(iorb,jorb,ik,iff)=tempf(iff)
        enddo
      enddo
    enddo
  enddo
  
end subroutine boson_mat_tau_to_freq


subroutine fderiv_dcmplx(m,n,x,f,g)
!     !INPUT/OUTPUT PARAMETERS:
!     m : order of derivative (in,integer)
!     n : number of points (in,integer)
!     x : abscissa array (in,real(n))
!     f : function array (in,complex(n))
!     g : (anti-)derivative of f (out,complex(n))
!     !DESCRIPTION:
!     Given function $f$ defined on a set of points $x_i$ then if $m\ge 0$ this
!     routine computes the $m$th derivative of $f$ at each point. If $m<0$ the
!     anti-derivative of $f$ given by
!     $$ g(x_i)=\int_{x_1}^{x_i} f(x)\,dx $$
!     is calculated. If $m=-1$ then an accurate integral is computed by fitting
!     the function to a clamped cubic spline_dcmplx. When $m=-3$ the fast but low
!     accuracy trapezoidal integration method is used. Simpson's integration,
!     which is slower but more accurate than the trapezoidal method, is used if
!     $m=-2$.
!     
!     !REVISION HISTORY:
!     Created May 2002 (JKD)
!     EOP
!     BOC
  implicit none
!     arguments
  integer, intent(in) :: m,n
  double precision, intent(in) :: x(n)
  complex*16, intent(in) :: f(n)
  complex*16, intent(out) :: g(n)
!     local variables
  integer :: i
  double precision ::  x0,x1,x2,dx
!     automatic arrays
  complex*16 :: cf(3,n)
! if (n.le.0) then
! write(*,*)
! write(*,'("Error(fderiv_dcmplx): invalid number of points : ",I8)') n
! write(*,*)
! stop
! end if
  select case(m)
  case(-3)
!     low accuracy trapezoidal integration
    g(1)=0.d0
    do i=1,n-1
      g(i+1)=g(i)+0.5d0*(x(i+1)-x(i))*(f(i+1)+f(i))
    end do
    return
  case(-2)
!     medium accuracy Simpson integration
    g(1)=0.d0
    do i=1,n-2
      x0=x(i)
      x1=x(i+1)
      x2=x(i+2)
      g(i+1)=g(i)+(x0-x1)*(f(i+2)*(x0-x1)**2 &
        +f(i+1)*(x2-x0)*(x0+2.d0*x1-3.d0*x2) &
        +f(i)*(x2-x1)*(2.d0*x0+x1-3.d0*x2))/(6.d0*(x0-x2)*(x1-x2))
    end do
    x0=x(n)
    x1=x(n-1)
    x2=x(n-2)
    g(n)=g(n-1)+(x1-x0)*(f(n-2)*(x1-x0)**2 &
      +f(n)*(x1-x2)*(3.d0*x2-x1-2.d0*x0)&
      +f(n-1)*(x0-x2)*(3.d0*x2-2.d0*x1-x0))/(6.d0*(x2-x1)*(x2-x0))
    return
  case(0)
    g(:)=f(:)
    return
  case(4:)
    g(:)=0.d0
    return
  end select
!     high accuracy integration/differentiation from spline_dcmplx interpolation
  call spline_dcmplx(n,x,f,cf)
  select case(m)
  case(:-1)
    g(1)=0.d0
    do i=1,n-1
      dx=x(i+1)-x(i)
      g(i+1)=g(i)+(((0.25d0*cf(3,i)*dx &
        +0.3333333333333333333d0*cf(2,i))*dx &
        +0.5d0*cf(1,i))*dx+f(i))*dx
    end do
  case(1)
    g(:)=cf(1,:)
  case(2)
    g(:)=2.d0*cf(2,:)
  case(3)
    g(:)=6.d0*cf(3,:)
  end select
  return
end subroutine fderiv_dcmplx


!     Copyright (C) 2011 J. K. Dewhurst, S. Sharma and E. K. U. Gross.
!     This file is distributed under the terms of the GNU Lesser General Public
!     This file is modified to be suitable for complex variables by sangkook choi  !     License. See the file COPYING for license details.

!     BOP
!     !ROUTINE: spline_dcmplx
!     !INTERFACE:
subroutine spline_dcmplx(n,x,f,cf)
!     !INPUT/OUTPUT PARAMETERS:
!     n  : number of points (in,integer)
!     x  : abscissa array (in,real(n))
!     f  : input data array (in,complex*16(n))
!     cf : cubic spline_dcmplx coefficients (out,complex*16(3,n))
!     !DESCRIPTION:
!     Calculates the coefficients of a cubic spline_dcmplx fitted to input data. In other
!     words, given a set of data points $f_i$ defined at $x_i$, where
!     $i=1\ldots n$, the coefficients $c_j^i$ are determined such that
!     $$ y_i(x)=f_i+c_1^i(x-x_i)+c_2^i(x-x_i)^2+c_3^i(x-x_i)^3, $$
!     is the interpolating function for $x\in[x_i,x_{i+1})$. The coefficients are
!     determined piecewise by fitting a cubic polynomial to adjacent points.
!     
!     !REVISION HISTORY:
!     Created November 2011 (JKD)
!     EOP
!     BOC
  implicit none
!     arguments
  integer, intent(in) :: n
  double precision, intent(in) :: x(n)
  complex*16, intent(in) :: f(n)
  complex*16, intent(out) :: cf(3,n)
!     local variables
  integer ::  i
  double precision :: x0,x1,x2,x3
  complex*16 :: y0,y1,y2,y3,c1,c2,c3,t0,t1,t2,t3,t4,t5,t6
! c$$$  if (n.le.0) then
! c$$$  write(*,*)
! c$$$  write(*,'("Error(spline_dcmplx): n <= 0 : ",I8)') n
! c$$$  write(*,*)
! c$$$  stop
! c$$$  end if
  if (n.eq.1) then
    cf(:,1)=0.d0
    return
  end if
  if (n.eq.2) then
    cf(1,1)=(f(2)-f(1))/(x(2)-x(1))
    cf(2:3,1)=0.d0
    cf(1,2)=cf(1,1)
    cf(2:3,2)=0.d0
    return
  end if
  if (n.eq.3) then
    x0=x(1)
    x1=x(2)-x0
    x2=x(3)-x0
    y0=f(1)
    y1=f(2)-y0
    y2=f(3)-y0
    t0=1.d0/(x1*x2*(x2-x1))
    t1=x1*y2
    t2=x2*y1
    c1=t0*(x2*t2-x1*t1)
    c2=t0*(t1-t2)
    cf(1,1)=c1
    cf(2,1)=c2
    cf(3,1)=0.d0
    t3=2.d0*c2
    cf(1,2)=c1+t3*x1
    cf(2,2)=c2
    cf(3,2)=0.d0
    cf(1,3)=c1+t3*x2
    cf(2,3)=c2
    cf(3,3)=0.d0
    return
  end if
  y0=f(1)
  y1=f(2)-y0
  y2=f(3)-y0
  y3=f(4)-y0
  x0=x(1)
  x1=x(2)-x0
  x2=x(3)-x0
  x3=x(4)-x0
  t0=1.d0/(x1*x2*x3*(x1-x2)*(x1-x3)*(x2-x3))
  t1=x1*x2*y3
  t2=x2*x3*y1
  t3=x3*x1*y2
  t4=x1**2
  t5=x2**2
  t6=x3**2
  y1=t3*t6-t1*t5
  y3=t2*t5-t3*t4
  y2=t1*t4-t2*t6
  c1=t0*(x1*y1+x2*y2+x3*y3)
  c2=-t0*(y1+y2+y3)
  c3=t0*(t1*(x1-x2)+t2*(x2-x3)+t3*(x3-x1))
  cf(1,1)=c1
  cf(2,1)=c2
  cf(3,1)=c3
  cf(1,2)=c1+2.d0*c2*x1+3.d0*c3*t4
  cf(2,2)=c2+3.d0*c3*x1
  cf(3,2)=c3
  if (n.eq.4) then
    cf(1,3)=c1+2.d0*c2*x2+3.d0*c3*t5
    cf(2,3)=c2+3.d0*c3*x2
    cf(3,3)=c3
    cf(1,4)=c1+2.d0*c2*x3+3.d0*c3*t6
    cf(2,4)=c2+3.d0*c3*x3
    cf(3,4)=c3
    return
  end if
  do i=3,n-2
    y0=f(i)
    y1=f(i-1)-y0
    y2=f(i+1)-y0
    y3=f(i+2)-y0
    x0=x(i)
    x1=x(i-1)-x0
    x2=x(i+1)-x0
    x3=x(i+2)-x0
    t1=x1*x2*y3
    t2=x2*x3*y1
    t3=x3*x1*y2
    t0=1.d0/(x1*x2*x3*(x1-x2)*(x1-x3)*(x2-x3))
    c3=t0*(t1*(x1-x2)+t2*(x2-x3)+t3*(x3-x1))
    t4=x1**2
    t5=x2**2
    t6=x3**2
    y1=t3*t6-t1*t5
    y2=t1*t4-t2*t6
    y3=t2*t5-t3*t4
    cf(1,i)=t0*(x1*y1+x2*y2+x3*y3)
    cf(2,i)=-t0*(y1+y2+y3)
    cf(3,i)=c3
  end do
  c1=cf(1,n-2)
  c2=cf(2,n-2)
  c3=cf(3,n-2)
  cf(1,n-1)=c1+2.d0*c2*x2+3.d0*c3*t5
  cf(2,n-1)=c2+3.d0*c3*x2
  cf(3,n-1)=c3
  cf(1,n)=c1+2.d0*c2*x3+3.d0*c3*t6
  cf(2,n)=c2+3.d0*c3*x3
  cf(3,n)=c3
  return
end subroutine spline_dcmplx





subroutine cal_u_matrix(norb,is_spinorbit,slaterf,ad_trans,umatrix)
  implicit none
  integer, intent(in) :: norb, is_spinorbit
  double precision,intent(in) :: slaterf(0:3)
  complex*16, intent(in) :: ad_trans(norb,norb)
  complex*16,intent(out) :: umatrix(norb,norb,norb,norb)


  integer :: kk, ll, lval,norb_2
  complex*16, allocatable ::  &
    rotmat_cmplx2real(:,:), &
    rotmat_new(:,:), &
    umatrix_temp(:,:,:,:), &
    rotmat_mlms2jmj(:,:)      

  umatrix=0.0d0            

  if  (is_spinorbit .eq. 0)then
    lval=(norb-1)/2
    call cal_coulomb_matrix_from_slater(lval,slaterf,umatrix)

    allocate(rotmat_cmplx2real(norb,norb))
    rotmat_cmplx2real=0.0d0
    call cal_rotmat_cmplx2real(norb,rotmat_cmplx2real)

    allocate(rotmat_new(norb,norb))
    rotmat_new=0.0d0

    call zgemm('n','n',norb,norb,norb,(1.d0,0.d0), &
      ad_trans,norb,rotmat_cmplx2real,norb, &
      (0.d0,0.d0),rotmat_new,norb)

    call rotate_umatrix(norb,umatrix,rotmat_new,0)

    deallocate(rotmat_cmplx2real)
    deallocate(rotmat_new)
  else
!     for wannier90 convension
!        j*2     5 5 5 5 5 5  7 7 7 7  7  7  7  7
!        mj*2   -5-3-1 1 3 5 -7-5-3-1  1  3  5  7
! default index  1 2 3 4 5 6  7 8 9 10 11 12 13 14        
    lval=(norb/2-1)/2
    norb_2=norb/2
    allocate(umatrix_temp(norb_2,norb_2,norb_2,norb_2))
    call cal_coulomb_matrix_from_slater(lval,slaterf,umatrix_temp)
    do kk=1,2
      do ll=1,2
        umatrix( &
          (1+(kk-1)*norb_2):(norb_2+(kk-1)*norb_2), &
          (1+(kk-1)*norb_2):(norb_2+(kk-1)*norb_2), &
          (1+(ll-1)*norb_2):(norb_2+(ll-1)*norb_2), &
          (1+(ll-1)*norb_2):(norb_2+(ll-1)*norb_2) &
          ) &
          =umatrix_temp
      enddo
    enddo

    allocate(rotmat_mlms2jmj(norb,norb))
    rotmat_mlms2jmj=0.0d0

    call cal_rotmat_mlms2jmj(lval, rotmat_mlms2jmj)
    allocate(rotmat_new(norb,norb))
    rotmat_new=0.0d0
    call zgemm('n','n',norb,norb, &
      norb,(1.d0,0.d0), &
      ad_trans,norb, &
      rotmat_mlms2jmj,norb, &
      (0.d0,0.d0), &
      rotmat_new,norb)        

    call rotate_umatrix(norb, umatrix, rotmat_new,0)
    deallocate(umatrix_temp)
    deallocate(rotmat_mlms2jmj)
    deallocate(rotmat_new)

  endif
end subroutine cal_u_matrix


subroutine cal_coulomb_matrix_from_slater(lval, ff, mat)
  implicit none
  integer, intent(in) :: lval
  double precision, intent(in) :: ff(0:3)
  complex*16, intent(out) :: mat &
    (2*lval+1,2*lval+1,2*lval+1,2*lval+1)      

  integer :: i1,i2,i3,i4,kk0,kk,mi1,mi3,qq,mq
  double precision, external :: f3j_int

  mat=0.0d0

  do kk=0,2*lval,2
    do qq=-kk, kk
      do i1=-lval, lval
        do i2=-lval, lval
          do i3=-lval, lval
            do i4=-lval, lval                  
              mq=-qq
              mi1=-i1
              mi3=-i3
              kk0=kk/2
              mat(i1+lval+1,i2+lval+1,i3+lval+1,i4+lval+1) &
                =mat(i1+lval+1,i2+lval+1,i3+lval+1,i4+lval+1) &
                +(2.0d0*lval+1.0d0)**2*ff(kk0) &
                *f3j_int(lval,0,kk,0,lval,0)**2 &
                *f3j_int(lval,mi1,kk,qq,lval,i2) &
                *f3j_int(lval,mi3,kk,mq,lval,i4) &
                *(-1)**(i1+i4)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

end subroutine cal_coulomb_matrix_from_slater



subroutine boson_den_spin_to_bilinear(lval,matin, matout)
  implicit none
  integer, intent(in) :: lval
  complex*16, intent(in) :: matin(4*lval+2, 4*lval+2)
  complex*16, intent(out) :: matout(4*lval*2+6*lval+2,4*lval*2+6*lval+2)


  integer :: is, dimin,dimout,mm,kk,qq,iorb,ind
  complex*16 :: transright(4*lval+2, 4*lval*2+6*lval+2), &
    transleft(4*lval*2+6*lval+2,4*lval+2), &
    tempmat(4*lval+2, 4*lval*2+6*lval+2)
  double precision, external :: gaunt

  dimin=(2*lval+1)*2
  dimout=(2*lval*2+3*lval+1)*2
  transleft=0.0d0
  transright=0.0d0  
  do mm=-lval, lval
    iorb=mm+lval+1
    ind=0
    do kk=0, 2*lval, 2
      do qq=-kk, kk
        ind=ind+1
        transright(iorb,ind)=gaunt(lval,mm,mm,kk,qq)
        transright(iorb+dimin/2,ind+dimout/2)=gaunt(lval,mm,mm,kk,qq)
        transleft(ind,iorb)=gaunt(lval,mm,mm,kk,qq)
        transleft(ind+dimout/2,iorb+dimin/2)=gaunt(lval,mm,mm,kk,qq)                  
      enddo
    enddo
  enddo

  call zgemm('n','n',dimin, dimout, dimin,(1.d0,0.d0), &
    matin,dimin,transright,dimin,(0.d0,0.d0),tempmat,dimin)
  call zgemm('n','n',dimout, dimout, dimin,(1.d0,0.d0), &
    transleft,dimout,tempmat,dimin,(0.d0,0.d0),matout,dimout)

end subroutine boson_den_spin_to_bilinear


subroutine boson_den_to_bilinear(lval,matin, matout)
  implicit none
  integer, intent(in) :: lval
  complex*16, intent(in) :: matin(2*lval+1, 2*lval+1)
  complex*16, intent(out) :: matout(2*lval*2+3*lval+1,2*lval*2+3*lval+1)


  integer :: is, dimin,dimout,mm,kk,qq,iorb,ind
  complex*16 :: transright(2*lval+1, 2*lval*2+3*lval+1), &
    transleft(2*lval*2+3*lval+1,2*lval+1), &
    tempmat(2*lval+1, 2*lval*2+3*lval+1)
  double precision, external :: gaunt

  dimin=(2*lval+1)
  dimout=(2*lval*2+3*lval+1)
  transleft=0.0d0
  transright=0.0d0  
  do mm=-lval, lval
    iorb=mm+lval+1
    ind=0
    do kk=0, 2*lval, 2
      do qq=-kk, kk
        ind=ind+1
        transright(iorb,ind)=gaunt(lval,mm,mm,kk,qq)
        transleft(ind,iorb)=gaunt(lval,mm,mm,kk,qq)
      enddo
    enddo
  enddo

  call zgemm('n','n',dimin, dimout, dimin,(1.d0,0.d0), &
    matin,dimin,transright,dimin,(0.d0,0.d0),tempmat,dimin)
  call zgemm('n','n',dimout, dimout, dimin,(1.d0,0.d0), &
    transleft,dimout,tempmat,dimin,(0.d0,0.d0),matout,dimout)

end subroutine boson_den_to_bilinear





DOUBLE PRECISION function gaunt(lval,m1,m2,kk,qq)
  implicit none
  integer, intent(in) :: lval, m1, m2, kk, qq

  double precision :: pi
  double precision, external :: f3j_int

  pi=datan2(1.0d0,1.0d0)*4.0d0
  gaunt=(2*lval+1)*dsqrt((2*kk+1)/4.0d0/pi)*(-1)**(-m1)*f3j_int(lval,0,kk,0,lval,0)*f3j_int(lval,-m1,kk,qq,lval,m2)
end function gaunt



DOUBLE PRECISION function f3j_int(j1, m1, j2, m2, j3, m3)
!  wigner 3j symbol 
IMPLICIT NONE
integer, intent(in) :: j1, j2, j3, m1, m2, m3
INTEGER            :: tmin, tmax, t
DOUBLE PRECISION             :: sum, v1, v2, dn, &
  j1d,j2d,j3d,m1d,m2d,m3d
!     function calls
DOUBLE PRECISION,external :: dFactorial
DOUBLE PRECISION,external :: iFactorial
DOUBLE PRECISION,external :: Ddelta
DOUBLE PRECISION,external :: mone
f3j_int=0
j1d=j1*1.0d0
j2d=j2*1.0d0
j3d=j3*1.0d0
m1d=m1*1.0d0
m2d=m2*1.0d0
m3d=m3*1.0d0
IF (abs(m1d+m2d+m3d) .GT. 1e-10) return
IF (abs(j1d-j2d)-1e-14.GT.j3d.OR.j3d.GT.j1d+j2d+1e-14) return
if (abs(m1d).GT.j1d.OR.abs(m2d).GT.j2d.OR.abs(m3d).GT.j3d) &
  return
tmin = INT(max(max(0.0,j2d-j3d-m1d),j1d-j3d+m2d)+1e-14)
tmax = INT(min(min(j1d+j2d-j3d,j1d-m1d),j2d+m2d)+1e-14)
sum=0
DO t=tmin, tmax
  v1 = dFactorial(j3d-j2d+m1d+t)*dFactorial(j3d-j1d-m2d+t)
  v2 = dFactorial(j1d+j2d-j3d-t)*dFactorial(j1d-m1d-t) &
    *dFactorial(j2d+m2d-t)
  sum = sum + mone(t)/(iFactorial(t)*v1*v2)
END DO
dn = dFactorial(j1d+m1d)*dFactorial(j1d-m1d)*dFactorial(j2d+m2d) &
  *dFactorial(j2d-m2d)*dFactorial(j3d+m3d)*dFactorial(j3d-m3d)
f3j_int = mone(INT(j1d-j2d-m3d))*Ddelta(j1d,j2d,j3d)*sqrt(dn)*sum
return
END function f3j_int



DOUBLE PRECISION function iFactorial(j)
IMPLICIT NONE
INTEGER, intent(in) :: j
INTEGER :: i
DOUBLE PRECISION :: x
if (j<0) print *, &
  "iFactorial defined only for non-negative numbers!"
x=1
iFactorial = x
if (j.eq.1) return
DO i=2,j
  x = x*i
END DO
iFactorial = x
return
end function iFactorial

DOUBLE PRECISION function dFactorial(x)
IMPLICIT NONE
DOUBLE PRECISION, intent(in) :: x
DOUBLE PRECISION, PARAMETER :: spi2 = 0.8862269254527579
DOUBLE PRECISION :: y, r
r=1
y=x
DO WHILE(y.gt.1.0)
  r= r * y
  y= y -1.
ENDDO
IF (abs(y-0.5).LT.1e-10) r = r*spi2
dFactorial = r
return
END function dFactorial

DOUBLE PRECISION function mone(i)
INTEGER, intent(in) :: i
mone = 1 - 2*MOD(abs(i),2)
return
end function mone

DOUBLE PRECISION function Ddelta(j1, j2, j)
IMPLICIT NONE
DOUBLE PRECISION, intent(in) :: j1, j2, j
!     function calls
DOUBLE PRECISION :: dFactorial
Ddelta = sqrt(dFactorial(j1+j2-j)*dFactorial(j1-j2+j) &
  *dFactorial(-j1+j2+j)/dFactorial(j1+j2+j+1))
return
END function Ddelta


subroutine cal_rotmat_mlms2jmj(ll, rotmat)

implicit none
integer, intent(in) :: ll
complex*16, intent(out) :: rotmat(4*ll+2,4*ll+2)

integer:: jj2,indms,indml,indmlms,indi,indmj,indjmj, &
  iorb,jorb
double precision :: jj,ulmu,fac1,fac2

!     default order:
!     mj=> fastest index, starting from smallest,
!     i=>next fastest, starting from smallest,

!     ml=> fastest index, starting from smallest,
!     ms=>next fastest, starting from smallest,            

do indms=-1,1,2      
  do indml=-ll, ll
    indmlms=indml+ll+1+(indms+1)/2*(2*ll+1)
    do indi=-1,1,2
      jj2=2*ll+indi
      do indmj=-jj2, jj2, 2
        indjmj=(indmj+jj2)/2+1+(indi+1)/2*(jj2-1)
        ulmu=dble(indmj)/2.0d0/(dble(ll)+0.5d0)
        if (indi .eq. indms) then
          fac1=1.0d0
        else
          fac1=-1.0d0
        endif
        fac2=1.0d0
        if (indi .eq. -1 .and. indms .eq. 1) then
          fac2=-1.0d0
        endif
        if (indmj .eq. indms+indml*2) then
          rotmat(indjmj,indmlms)=1.0d0/dsqrt(2.0d0) &
            *dsqrt(1+ulmu*fac1)*fac2
        endif
      enddo
    enddo
  enddo
enddo


end subroutine cal_rotmat_mlms2jmj


subroutine cal_rotmat_cmplx2real(norb, rotmat)
implicit none
integer, intent(in) :: norb
complex*16, intent(out) :: rotmat(norb,norb)

integer :: iorb,jorb

rotmat=0.0d0

rotmat(norb/2+1,norb/2+1)=1.0d0       
do iorb=1, norb/2
  rotmat(iorb+norb/2+1,iorb+norb/2+1) &
    =1.0d0/dsqrt(2.0d0)*(-1)**iorb
  rotmat(iorb+norb/2+1,-iorb+norb/2+1) &
    =1.0d0/dsqrt(2.0d0)
  rotmat(-iorb+norb/2+1,iorb+norb/2+1) &
    =1.0d0/dsqrt(2.0d0)*dcmplx(0.0d0, 1.0d0)*(-1)**(iorb+1)
  rotmat(-iorb+norb/2+1,-iorb+norb/2+1) &
    =1.0d0/dsqrt(2.0d0)*dcmplx(0.0d0, 1.0d0)
enddo
end subroutine cal_rotmat_cmplx2real


subroutine rotate_umatrix(norb,umatrix, rotmat,daggerflag)
implicit none

integer, intent(in) :: norb,daggerflag
complex*16, intent(inout) :: umatrix(norb,norb,norb,norb)
complex*16, intent(in) :: rotmat(norb,norb)


integer :: iorb1,iorb2,jorb1,jorb2,korb1,korb2,lorb1,lorb2, &
  iorb,jorb,korb,lorb 
complex*16 :: umatrix_temp(norb,norb,norb,norb), &
  rotmat_temp(norb,norb)

if (daggerflag .eq. 1) then
  rotmat_temp=dconjg(transpose(rotmat))
else
  rotmat_temp=rotmat
endif


umatrix_temp=umatrix
umatrix=0.0d0

do lorb2=1, norb
  do korb2=1, norb
    do jorb2=1, norb
      do iorb2=1, norb            
        do iorb1=1, norb
          umatrix(iorb1,jorb2,korb2,lorb2) &
            =umatrix(iorb1,jorb2,korb2,lorb2) &
            +rotmat_temp(iorb1,iorb2) &
            *umatrix_temp(iorb2,jorb2,korb2,lorb2)
        enddo

      enddo
    enddo
  enddo
enddo

umatrix_temp=umatrix
umatrix=0.0d0

do lorb2=1, norb
  do korb2=1, norb
    do jorb2=1, norb
      do iorb2=1, norb            
        do jorb1=1, norb            
          umatrix(iorb2,jorb1,korb2,lorb2) &
            =umatrix(iorb2,jorb1,korb2,lorb2) &
            +dconjg(rotmat_temp(jorb1,jorb2)) &
            *umatrix_temp(iorb2,jorb2,korb2,lorb2)
        enddo

      enddo
    enddo
  enddo
enddo


umatrix_temp=umatrix      
umatrix=0.0d0
do lorb2=1, norb
  do korb2=1, norb
    do jorb2=1, norb
      do iorb2=1, norb            
        do korb1=1, norb                        
          umatrix(iorb2,jorb2,korb1,lorb2) &
            =umatrix(iorb2,jorb2,korb1,lorb2) &
            +rotmat_temp(korb1,korb2) &
            *umatrix_temp(iorb2,jorb2,korb2,lorb2)
        enddo

      enddo
    enddo
  enddo
enddo


umatrix_temp=umatrix      
umatrix=0.0d0
do lorb2=1, norb
  do korb2=1, norb
    do jorb2=1, norb
      do iorb2=1, norb            
        do lorb1=1, norb                        
          umatrix(iorb2,jorb2,korb2,lorb1) &
            =umatrix(iorb2,jorb2,korb2,lorb1) &
            +dconjg(rotmat_temp(lorb1,lorb2)) &
            *umatrix_temp(iorb2,jorb2,korb2,lorb2)
        enddo

      enddo
    enddo
  enddo
enddo


end subroutine rotate_umatrix

subroutine cal_chi(norb,pola_nu,umat,chi)
implicit none

integer, intent(in) :: norb
complex*16,intent(in) :: pola_nu(norb,norb,norb,norb), umat(norb,norb,norb,norb)
complex*16, intent(out) :: chi(norb,norb,norb,norb)


integer :: tau1,tau2
complex*16 :: epsilonmat(norb,norb,norb,norb),epsiloninv(norb,norb,norb,norb)


epsilonmat=0.0d0
epsiloninv=0.0d0
chi=0.0d0

call zgemm('n','n',norb**2,norb**2, &
  norb**2,(-1.0d0,0.0d0),&
  umat,norb**2, &
  pola_nu,norb**2, &
  (0.0d0,0.0d0),epsilonmat,norb**2)
do tau1=1, norb
  do tau2=1,norb             
    epsilonmat(tau1,tau2,tau1,tau2) &
      =epsilonmat(tau1,tau2,tau1,tau2)+1.0d0
  enddo
enddo
call dcmplx_matinv(epsilonmat, &
  epsiloninv, &
  norb**2, norb**2)

call zgemm('n','n',norb**2,norb**2, &
  norb**2,(1.0d0,0.0d0), &
  pola_nu,norb**2, &
  epsiloninv,norb**2, &
  (0.0d0,0.0d0),chi,norb**2)

end subroutine cal_chi

subroutine dcmplx_matinv(mat, invmat, dim,dimmax)
implicit none
integer, intent(in) :: dim, dimmax
complex*16, intent(in) :: mat(dimmax, dimmax)
complex*16, intent(out) :: invmat(dimmax,dimmax)

integer :: ipiv(dim),info
complex*16 :: wrk(dim**2)

invmat=mat
call zgetrf(dim,dim,invmat, &
  dimmax,ipiv,info)
! c$$$      if (info .ne. 0) then
! c$$$        write(179, *) 'error in zgetrf', info
! c$$$      endif
call zgetri(dim,invmat,dimmax, &
  ipiv,wrk,dim**2,info)          
! c$$$      if (info .ne. 0) then
! c$$$        write(179, *) 'error in zgetri', info
! c$$$      endif
end subroutine dcmplx_matinv
