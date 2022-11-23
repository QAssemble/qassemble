Module Fourier
  use Common
  implicit None

  private:: &
    Dyn_T2F,& 
    FLat_KR,BLat_KR


  Public :: &
! T2F, F2T, M, K2R, R2K
    
    ! FLatDyn_Normalization, & !
    ! FLocDyn_Normalization, &!
    
! interface T2F  
!    module procedure &
    FLatDyn_T2F, & !
    FLocDyn_T2F, &!
                                ! FLatDyn_T2F_v0, & !        
                                ! FLocDyn_T2F_v0, &!    
    
    BLatDyn_T2F, & !
    BLocDyn_T2F, & !
                                ! BLatDyn_T2F_v0, & !
                                ! BLocDyn_T2F_v0, & !
    
! end interface T2F
    
! interface F2T 
!    module procedure &
    FLatDyn_F2T, & !
                                ! FLatDyn_F2T_v0, & !    
    FLocDyn_F2T, & !
                                ! FLocDyn_F2T_v0, &!
    BLatDyn_F2T,& !
                                ! BLatDyn_F2T_v0,& !    
    BLocDyn_F2T,& !
                                ! BLocDyn_F2T_v0,& !          
    
! end interface F2T
    
! interface M 
!    module procedure &
    FLatDyn_M, & !
    FLocDyn_M, & !
    BLatDyn_M, & !
    BLocDyn_M, & !
! end interface M
    
! interface K2R
!    module procedure &
    FLatDyn_K2R, &
    BLatDyn_K2R, & 
    FLatStc_K2R, &
    BLatStc_K2R, &
! end interface K2R
    
! interface R2K
!    module procedure &
    FLatDyn_R2K, &
    BLatDyn_R2K, &
    FLatStc_R2K, &
    BLatStc_R2K
! end interface R2K

contains


  subroutine FLocDyn_F2T(norb,ns,nomega,omega,fomega,moment,ntau,tau,ftau)
    implicit none

    integer, intent(in) :: nomega,ntau,norb,ns
    double precision, intent(in) :: tau(0:(ntau-1)),omega(0:(nomega-1))
    complex*16, intent(in) :: fomega(norb,norb,ns,0:(nomega-1)),moment(norb,norb,ns,3)
    complex*16, intent(out) :: ftau(norb,norb,ns,0:(ntau-1))

    integer :: itau,iomega, ii,iorb,jorb,is
    double precision :: beta, pi, xx


    integer :: ierr
    integer*8 :: ntau_finu, nf_finu

    double precision ::taurad_finu(0:(ntau-1))
    complex*16 :: ff_finu((-2*nomega+1):(2*nomega-1)), ftau_finu(0:(ntau-1)), &
      momega_finu((-2*nomega+1):(2*nomega-1),3), mtau_finu(0:(ntau-1),3), ai

    integer*8, allocatable :: null

    ai=dcmplx(0.0d0, 1.0d0)
    ntau_finu=ntau
    nf_finu=nomega*4-1

    pi=datan2(1.0d0,1.0d0)*4.0d0

! beta=tau(0)*2.0d0/(dcos(pi*(ntau-0.5d0)/dble(ntau))+1.0d0)
! print *, 'beta', beta
    beta=pi/omega(0)

    momega_finu=0.0d0

    do iomega=-2*nomega+1, 2*nomega-1
      if (modulo(iomega, 2) .eq. 1) then
        momega_finu(iomega,1)=1.0d0/(pi/beta*iomega*ai)
        momega_finu(iomega,2)=1.0d0/(pi/beta*iomega*ai)**2
        momega_finu(iomega,3)=1.0d0/(pi/beta*iomega*ai)**3
      endif
    enddo


    taurad_finu=0.0d0

    do itau=0, ntau-1
      taurad_finu(itau)=tau(itau)/beta*pi
    enddo


    mtau_finu=0.0d0

    do ii=1, 3
      call finufft1d2(ntau_finu,taurad_finu,mtau_finu(:,ii),-1, 1.0d-12, nf_finu,momega_finu(:,ii),null,ierr)
    enddo


    ftau=0.0d0

    do is=1, ns
      do iorb=1, norb
        do jorb=1, norb
          ff_finu=0.0d0
          ftau_finu=0.0d0
          do iomega=-2*nomega+1, 2*nomega-1
            if (modulo(iomega, 2) .eq. 1) then
              if (iomega .gt. 0) then
                ff_finu(iomega)=fomega(iorb,jorb,is,(iomega-1)/2)
              else
                ff_finu(iomega)=dconjg(fomega(jorb,iorb,is,(-iomega-1)/2))
              endif
            endif
          enddo

          call finufft1d2(ntau_finu,taurad_finu,ftau_finu,-1, 1.0d-12, nf_finu,ff_finu,null,ierr)

          do itau=0, ntau-1
            xx=tau(itau)/beta                            
            ftau(iorb,jorb,is,itau)=ftau_finu(itau)/beta
            do ii=1, 3            
              ftau(iorb,jorb,is,itau) &
                =ftau(iorb,jorb,is,itau) &                
                -moment(iorb,jorb,is,ii)*mtau_finu(itau,ii)/beta &
                +1.0d0/2.0d0*beta**(ii-1)/factorial_int(ii-1)*(-1)**ii*eulerpolynomial(xx, ii-1)*moment(iorb,jorb,is,ii)
            enddo
          enddo
        enddo
      enddo
    enddo
  end subroutine FLocDyn_F2T


!   subroutine FLocDyn_F2T_v0(norb,ns,nomega,omega,fomega,moment,ntau,tau,ftau)
!     implicit none

!     integer, intent(in) :: nomega,ntau,norb,ns
!     double precision, intent(in) :: tau(0:(ntau-1)),omega(0:(nomega-1))
!     complex*16, intent(in) :: fomega(norb,norb,ns,0:(nomega-1)),moment(norb,norb,ns,3)
!     complex*16, intent(out) :: ftau(norb,norb,ns,0:(ntau-1))

!     integer :: itau,iomega, ii,iorb,jorb,is
!     double precision :: beta, pi, xx
!     complex*16 :: ai
! ! double precision, external :: factorial_int
! ! double precision, external :: eulerpolynomial


!     pi=datan2(1.0d0,1.0d0)*4.0d0
!     beta=pi/omega(0)
!     ai=dcmplx(0.0d0, 1.0d0)

!     ftau=0.0d0


!     do is=1, ns
!       do iorb=1, norb
!         do jorb=1, norb
!           do itau=0, ntau-1
!             do iomega=0, nomega-1
!               ftau(iorb,jorb,is,itau) &
!                 =ftau(iorb,jorb,is,itau) &
!                 +1.0d0/beta*cdexp(-tau(itau)*omega(iomega)*ai) &
!                 *( &
!                 fomega(iorb,jorb,is,iomega) & 
!                 -moment(iorb,jorb,is,1)/(omega(iomega)*ai) &
!                 -moment(iorb,jorb,is,2)/(omega(iomega)*ai)**2 &
!                 -moment(iorb,jorb,is,3)/(omega(iomega)*ai)**3 &                
!                 ) &
!                 +1.0d0/beta*cdexp(tau(itau)*omega(iomega)*ai) &
!                 *( &
!                 dconjg(fomega(jorb,iorb,is,iomega)) &
!                 +moment(iorb,jorb,is,1)/(omega(iomega)*ai) &
!                 -moment(iorb,jorb,is,2)/(omega(iomega)*ai)**2 &
!                 +moment(iorb,jorb,is,3)/(omega(iomega)*ai)**3 &                
!                 )
!             enddo

!             do ii=1, 3
!               xx=tau(itau)/beta                
!               ftau(iorb,jorb,is,itau) &
!                 =ftau(iorb,jorb,is,itau) &
!                 +1.0d0/2.0d0*beta**(ii-1)/factorial_int(ii-1)*(-1)**ii*eulerpolynomial(xx, ii-1)*moment(iorb,jorb,is,ii)          
!             enddo
!           enddo
!         enddo
!       enddo
!     enddo
!   end subroutine FLocDyn_F2T_V0


  subroutine FLatDyn_F2T(norb,ns,nk,nomega,omega,fomega,moment,ntau,tau,ftau)
    implicit none

    integer, intent(in) :: norb,ns,nomega,ntau,nk
    double precision, intent(in) :: tau(0:(ntau-1)), omega(0:(nomega-1))
    complex*16, intent(in) :: moment(norb,norb,ns,nk,3),fomega(norb,norb,ns,nk,0:(nomega-1))
    complex*16, intent(out) :: ftau(norb,norb,ns,nk,0:(ntau-1))

    integer :: ik,is

    ftau=0.0d0
    do ik=1, nk
      call FLocDyn_F2T(norb,ns,nomega,omega,fomega(:,:,:,ik,:),moment(:,:,:,ik,:),ntau,tau,ftau(:,:,:,ik,:))
    enddo
  end subroutine FLatDyn_F2T


! subroutine FLatDyn_F2T_v0(norb,ns,nk,nomega,omega,fomega,moment,ntau,tau,ftau)
!   implicit none

!   integer, intent(in) :: norb,ns,nomega,ntau,nk
!   double precision, intent(in) :: tau(0:(ntau-1)), omega(0:(nomega-1))
!   complex*16, intent(in) :: moment(norb,norb,ns,nk,3),fomega(norb,norb,ns,nk,0:(nomega-1))
!   complex*16, intent(out) :: ftau(norb,norb,ns,nk,0:(ntau-1))

!   integer :: ik,is

!   ftau=0.0d0
!   do ik=1, nk
!     call FLocDyn_F2T_v0(norb,ns,nomega,omega,fomega(:,:,:,ik,:),moment(:,:,:,ik,:),ntau,tau,ftau(:,:,:,ik,:))
!   enddo
! end subroutine FLatDyn_F2T_v0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine BLocDyn_F2T(norb,ns,nnu,nu,fnu,moment,ntau,tau,ftau)
    implicit none

    integer, intent(in) :: nnu,ntau,ns,norb
    double precision, intent(in) :: tau(0:(ntau-1)),nu(0:(nnu-1))
    complex*16, intent(in) :: fnu(norb,norb,ns,ns,0:(nnu-1)),moment(norb,norb,ns,ns,3)
    complex*16, intent(out) :: ftau(norb,norb,ns,ns,0:(ntau-1))

    integer :: itau,inu,iorb,jorb,is,js,ii
    double precision :: beta, pi,xx
    complex*16 :: ai

    integer :: ierr
    integer*8 :: ntau_finu, nf_finu

    double precision ::taurad_finu(0:(ntau-1))
    complex*16 :: ff_finu((-nnu+1):(nnu-1)), ftau_finu(0:(ntau-1)), &
      mnu_finu((-nnu+1):(nnu-1),3), mtau_finu(0:(ntau-1),3)

    integer*8, allocatable :: null    

    ai=dcmplx(0.0d0, 1.0d0)    
    pi=datan2(1.0d0,1.0d0)*4.0d0
    beta=2.0d0*pi/nu(1)

    ntau_finu=ntau
    nf_finu=nnu*2-1

    mnu_finu=0.0d0


    do inu=-nnu+1, nnu-1
      if (inu .ne. 0) then
        mnu_finu(inu,1)=1.0d0/(2.0d0*pi/beta*inu*ai)
        mnu_finu(inu,2)=1.0d0/(2.0d0*pi/beta*inu*ai)**2
        mnu_finu(inu,3)=1.0d0/(2.0d0*pi/beta*inu*ai)**3
      endif
    enddo

    taurad_finu=0.0d0

    do itau=0, ntau-1
      taurad_finu(itau)=tau(itau)/beta*2.0d0*pi
    enddo

    mtau_finu=0.0d0
    do ii=1, 3
      call finufft1d2(ntau_finu,taurad_finu,mtau_finu(:,ii),-1, 1.0d-12, nf_finu,mnu_finu(:,ii),null,ierr)
    enddo


    ftau=0.0d0
    do is=1, ns
      do js=1, ns      
        do iorb=1, norb
          do jorb=1, norb
            ff_finu=0.0d0
            ftau_finu=0.0d0
            do inu=-nnu+1, nnu-1
              if (inu .ge.0) then
                ff_finu(inu)=fnu(iorb,jorb,is,js,inu)
              else
                ff_finu(inu)=dconjg(fnu(jorb,iorb,js,is,-inu))
              endif
            enddo

            call finufft1d2(ntau_finu,taurad_finu,ftau_finu,-1, 1.0d-12, nf_finu,ff_finu,null,ierr)

            do itau=0, ntau-1
              xx=tau(itau)/beta                            
              ftau(iorb,jorb,is,js,itau)=ftau_finu(itau)/beta            
              do ii=1, 3
                ftau(iorb,jorb,is,js,itau) &
                  =ftau(iorb,jorb,is,js,itau) &
                  -moment(iorb,jorb,is,js,ii)*mtau_finu(itau,ii)/beta &                 
                  +moment(iorb,jorb,is,js,ii)*(beta)**(ii-1)*(-1)**(ii-1)/factorial_int(ii)*BernoulliPolynomial(xx, ii)      
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
  end subroutine BLocDyn_F2T

!   subroutine BLocDyn_F2T_v0(norb,ns,nnu,nu,fnu,moment,ntau,tau,ftau)
!     implicit none

!     integer, intent(in) :: nnu,ntau,ns,norb
!     double precision, intent(in) :: tau(0:(ntau-1)),nu(0:(nnu-1))
!     complex*16, intent(in) :: fnu(norb,norb,ns,ns,0:(nnu-1)),moment(norb,norb,ns,ns,3)
!     complex*16, intent(out) :: ftau(norb,norb,ns,ns,0:(ntau-1))

!     integer :: itau,inu,iorb,jorb,is,js,ii
!     double precision :: beta, pi,xx
!     complex*16 :: ai
! ! double precision, external :: factorial_int
! ! double precision, external :: BernoulliPolynomial


!     pi=datan2(1.0d0,1.0d0)*4.0d0
!     beta=2.0d0*pi/nu(1)
!     ftau=0.0d0
!     ai=dcmplx(0.0d0, 1.0d0)

!     do is=1, ns
!       do js=1, ns      
!         do iorb=1, norb
!           do jorb=1, norb
!             do itau=0, ntau-1
!               do inu=0, nnu-1
!                 if (inu .eq. 0) then
!                   ftau(iorb,jorb,is,js,itau)=ftau(iorb,jorb,is,js,itau)+1.0d0/beta*(fnu(iorb,jorb,is,js,inu))
!                 else
!                   ftau(iorb,jorb,is,js,itau) &
!                     =ftau(iorb,jorb,is,js,itau) &
!                     +1.0d0/beta*cdexp(-tau(itau)*nu(inu)*ai) &
!                     *( &
!                     fnu(iorb,jorb,is,js,inu) & 
!                     -moment(iorb,jorb,is,js,1)/(nu(inu)*ai) &
!                     -moment(iorb,jorb,is,js,2)/(nu(inu)*ai)**2 &
!                     -moment(iorb,jorb,is,js,3)/(nu(inu)*ai)**3 &                
!                     ) &
!                     +1.0d0/beta*cdexp(tau(itau)*nu(inu)*ai) &
!                     *( &
!                     dconjg(fnu(jorb,iorb,js,is,inu)) &
!                     +moment(iorb,jorb,is,js,1)/(nu(inu)*ai) &
!                     -moment(iorb,jorb,is,js,2)/(nu(inu)*ai)**2 &
!                     +moment(iorb,jorb,is,js,3)/(nu(inu)*ai)**3 &                
!                     )          
!                 endif
!               enddo
!               do ii=1, 3
!                 xx=tau(itau)/beta
!                 ftau(iorb,jorb,is,js,itau)=ftau(iorb,jorb,is,js,itau)+moment(iorb,jorb,is,js,ii)*(beta)**(ii-1)*(-1)**(ii-1)/factorial_int(ii)*BernoulliPolynomial(xx, ii)
!               enddo
!             enddo
!           enddo
!         enddo
!       enddo
!     enddo
!   end subroutine BLocDyn_F2T_V0

  subroutine BLatDyn_F2T(norb,ns,nk,nnu,nu,fnu,moment,ntau,tau,ftau)
    implicit none

    integer, intent(in) :: norb,ns,nnu,ntau,nk
    double precision, intent(in) :: tau(0:(ntau-1)), nu(0:(nnu-1))
    complex*16, intent(in) :: moment(norb,norb,ns,ns,nk,3),fnu(norb,norb,ns,ns,nk,0:(nnu-1))
    complex*16, intent(out) :: ftau(norb,norb,ns,ns,nk,0:(ntau-1))

    integer :: ik


    ftau=0.0d0
    do ik=1, nk
      call BLocDyn_F2T(norb,ns,nnu,nu,fnu(:,:,:,:,ik,:),moment(:,:,:,:,ik,:),ntau,tau,ftau(:,:,:,:,ik,:))
    enddo

  end subroutine BLatDyn_F2T


! subroutine BLatDyn_F2T_v0(norb,ns,nk,nnu,nu,fnu,moment,ntau,tau,ftau)
!   implicit none

!   integer, intent(in) :: norb,ns,nnu,ntau,nk
!   double precision, intent(in) :: tau(0:(ntau-1)), nu(0:(nnu-1))
!   complex*16, intent(in) :: moment(norb,norb,ns,ns,nk,3),fnu(norb,norb,ns,ns,nk,0:(nnu-1))
!   complex*16, intent(out) :: ftau(norb,norb,ns,ns,nk,0:(ntau-1))

!   integer :: ik


!   ftau=0.0d0
!   do ik=1, nk
!     call BLocDyn_F2T_v0(norb,ns,nnu,nu,fnu(:,:,:,:,ik,:),moment(:,:,:,:,ik,:),ntau,tau,ftau(:,:,:,:,ik,:))
!   enddo

! end subroutine BLatDyn_F2T_v0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine FLat_KR(norb,ns,nrk,rkgrid,fin,fout,sign,norm)

    implicit none

    integer, intent(in) :: norb,ns,nrk,rkgrid(3), sign
    double precision, intent(in) :: norm
    complex*16, intent(in) :: fin(norb,norb,ns,nrk)
    complex*16, intent(out) :: fout(norb,norb,ns,nrk)

    integer :: iorb, jorb,irk,is
    complex*16 :: tempmat(nrk)

    fout=0.0d0
    do is=1, ns
      do jorb=1, norb      
        do iorb=1, norb
          tempmat=0.0d0
          do irk=1, nrk
            tempmat(irk)=fin(iorb,jorb,is,irk)
          enddo
          call fftw3_3d(tempmat,rkgrid(1),rkgrid(2),rkgrid(3),sign)
          do irk=1, nrk
            fout(iorb,jorb,is,irk)=tempmat(irk)*norm
          enddo
        enddo
      enddo
    enddo
  end subroutine FLat_KR

  subroutine FLatStc_K2R(norb,ns,nrk,rkgrid,fin,fout)
    implicit none
    integer, intent(in) :: norb,ns,nrk,rkgrid(3)
    complex*16, intent(in) :: fin(norb,norb,ns,nrk)
    complex*16, intent(out) :: fout(norb,norb,ns,nrk)
    double precision :: norm
    norm=1.0d0/dble(nrk)
    call FLat_KR(norb,ns,nrk,rkgrid,fin,fout,1,norm)
  end subroutine FLatStc_K2R


  subroutine FLatStc_R2K(norb,ns,nrk,rkgrid,fin,fout)
    implicit none
    integer, intent(in) :: norb,ns,nrk,rkgrid(3)
    complex*16, intent(in) :: fin(norb,norb,ns,nrk)
    complex*16, intent(out) :: fout(norb,norb,ns,nrk)
    double precision :: norm
    norm=1.0d0
    call FLat_KR(norb,ns,nrk,rkgrid,fin,fout,-1,norm)
  end subroutine FLatStc_R2K


  subroutine FLatDyn_K2R(norb,ns,nrk,nto,rkgrid,fin,fout)
    implicit none
    integer, intent(in) :: norb,ns,nrk,nto,rkgrid(3)
    complex*16, intent(in) :: fin(norb,norb,ns,nrk,0:(nto-1))
    complex*16, intent(out) :: fout(norb,norb,ns,nrk,0:(nto-1))

    integer :: ito

    fout=0.0d0
    do ito=0, nto-1
      call FLatStc_K2R(norb,ns,nrk,rkgrid,fin(:,:,:,:,ito),fout(:,:,:,:,ito))
    enddo
  end subroutine FLatDyn_K2R


  subroutine FLatDyn_R2K(norb,ns,nrk,nto,rkgrid,fin,fout)
    implicit none
    integer, intent(in) :: norb,ns,nrk,nto,rkgrid(3)
    complex*16, intent(in) :: fin(norb,norb,ns,nrk,0:(nto-1))
    complex*16, intent(out) :: fout(norb,norb,ns,nrk,0:(nto-1))

    integer :: ito

    fout=0.0d0
    do ito=0, nto-1
      call FLatStc_R2K(norb,ns,nrk,rkgrid,fin(:,:,:,:,ito),fout(:,:,:,:,ito))
    enddo
  end subroutine FLatDyn_R2K


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine BLat_KR(norb,ns,nrk,rkgrid,fin,fout,sign,norm)

    implicit none

    integer, intent(in) :: norb,ns,nrk,rkgrid(3), sign
    double precision, intent(in) :: norm
    complex*16, intent(in) :: fin(norb,norb,ns,ns,nrk)
    complex*16, intent(out) :: fout(norb,norb,ns,ns,nrk)

    integer :: iorb, jorb,irk,is,js
    complex*16 :: tempmat(nrk)

    fout=0.0d0
    do js=1, ns
      do is=1, ns
        do jorb=1, norb      
          do iorb=1, norb
            tempmat=0.0d0
            do irk=1, nrk
              tempmat(irk)=fin(iorb,jorb,is,js,irk)
            enddo
            call fftw3_3d(tempmat,rkgrid(1),rkgrid(2),rkgrid(3),sign)
            do irk=1, nrk
              fout(iorb,jorb,is,js,irk)=tempmat(irk)*norm
            enddo
          enddo
        enddo
      enddo
    enddo
  end subroutine BLat_KR

  subroutine BLatStc_K2R(norb,ns,nrk,rkgrid,fin,fout)

    implicit none

    integer, intent(in) :: norb,ns,nrk,rkgrid(3)

    complex*16, intent(in) :: fin(norb,norb,ns,ns,nrk)
    complex*16, intent(out) :: fout(norb,norb,ns,ns,nrk)
    double precision :: norm
    norm=1.0d0/nrk

    call BLat_KR(norb,ns,nrk,rkgrid,fin,fout,1,norm)
  end subroutine BLatStc_K2R


  subroutine BLatStc_R2K(norb,ns,nrk,rkgrid,fin,fout)

    implicit none

    integer, intent(in) :: norb,ns,nrk,rkgrid(3)

    complex*16, intent(in) :: fin(norb,norb,ns,ns,nrk)
    complex*16, intent(out) :: fout(norb,norb,ns,ns,nrk)
    double precision :: norm
    norm=1.0d0
    call BLat_KR(norb,ns,nrk,rkgrid,fin,fout,-1,norm)
  end subroutine BLatStc_R2K


  subroutine BLatDyn_K2R(norb,ns,nrk,nto,rkgrid,fin,fout)
    implicit none
    integer, intent(in) :: norb,ns,nrk,nto,rkgrid(3)
    complex*16, intent(in) :: fin(norb,norb,ns,ns,nrk,0:(nto-1))
    complex*16, intent(out) :: fout(norb,norb,ns,ns,nrk,0:(nto-1))

    integer :: ito
! complex*16 :: fin_sub(norb,norb,ns,ns,nrk),fout_sub(norb,norb,ns,ns,nrk),fout_temp(norb,norb,ns,ns,nrk,0:(nto-1))

    fout=0.0d0
    do ito=0, nto-1
      call BLatStc_K2R(norb,ns,nrk,rkgrid,fin(:,:,:,:,:,ito),fout(:,:,:,:,:,ito))
    enddo
  end subroutine BLatDyn_K2R


  subroutine BLatDyn_R2K(norb,ns,nrk,nto,rkgrid,fin,fout)
    implicit none
    integer, intent(in) :: norb,ns,nrk,nto,rkgrid(3)
    complex*16, intent(in) :: fin(norb,norb,ns,ns,nrk,0:(nto-1))
    complex*16, intent(out) :: fout(norb,norb,ns,ns,nrk,0:(nto-1))

    integer :: ito

    fout=0.0d0
    do ito=0, nto-1
      call BLatStc_R2K(norb,ns,nrk,rkgrid,fin(:,:,:,:,:,ito),fout(:,:,:,:,:,ito))
    enddo
  end subroutine BLatDyn_R2K


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine FLocDyn_M(norb,ns,nomega,omega,ff,isgreen,highzero, moment,high)
    implicit none
    integer, intent(in) :: norb, ns,nomega,isgreen,highzero
    double precision, intent(in) :: omega(0:(nomega-1))
    complex*16, intent(in) :: ff(norb,norb,ns,0:(nomega-1))

    complex*16,intent(out) :: moment(norb,norb,ns,3), high(norb,norb,ns)

    integer :: iorb, jorb, is, info,ii
    complex*16 :: ai
    integer, allocatable :: ipiv(:)
    complex*16, allocatable :: amat(:,:), bmat(:,:)

    moment=0.0d0
    high=0.0d0
    ai=dcmplx(0.0d0, 1.0d0)

    if (isgreen .eq. 1) then

      do is=1, ns
        do iorb=1, norb
          do jorb=1, norb

            if (iorb .eq. jorb) then
              moment(iorb,jorb,is,1)=1.0d0
            else
              moment(iorb,jorb,is,1)=0.0d0
            endif

            moment(iorb,jorb,is,2) &
              =moment(iorb,jorb,is,2) &
              +(ff(iorb,jorb,is,nomega-1)+dconjg(ff(jorb,iorb,is,nomega-1))) &
              /2.0d0*(omega(nomega-1)*ai)**2

            moment(iorb,jorb,is,3) &
              =moment(iorb,jorb,is,3) &
              +(ff(iorb,jorb,is,nomega-1)-dconjg(ff(jorb,iorb,is,nomega-1))-moment(iorb,jorb,is,1)*2.0d0/(omega(nomega-1)*ai))/2.0d0*(omega(nomega-1)*ai)**3                                        

          enddo
        enddo
      enddo

    else
      if (highzero .eq. 1) then
        do is=1, ns
          do iorb=1, norb
            do jorb=1, norb

              moment(iorb,jorb,is,1) &
                =moment(iorb,jorb,is,1) &
                +(ff(iorb,jorb,is,nomega-1)-dconjg(ff(jorb,iorb,is,nomega-1))) &
                /2.0d0*(omega(nomega-1)*ai)

              moment(iorb,jorb,is,2) &
                =moment(iorb,jorb,is,2) &
                +(ff(iorb,jorb,is,nomega-1)+dconjg(ff(jorb,iorb,is,nomega-1))) &
                /2.0d0*(omega(nomega-1)*ai)**2

            enddo
          enddo
        enddo
      else
        allocate(amat(4,4))
        allocate(bmat(4,1))
        allocate(ipiv(4))
        do is=1, ns
          do iorb=1, norb
            do jorb=1, norb
              amat=0.0d0
              bmat=0.0d0

              amat(1,:)=(/dcmplx(1.0d0,0.0d0), 1.0d0/(omega(nomega-1)*ai), 1.0d0/(omega(nomega-1)*ai)**2, 1.0d0/(omega(nomega-1)*ai)**3/)
              amat(2,:)=(/dcmplx(1.0d0,0.0d0), -1.0d0/(omega(nomega-1)*ai), 1.0d0/(omega(nomega-1)*ai)**2, -1.0d0/(omega(nomega-1)*ai)**3/)          
              amat(3,:)=(/dcmplx(1.0d0,0.0d0), 1.0d0/(omega(nomega-2)*ai), 1.0d0/(omega(nomega-2)*ai)**2, 1.0d0/(omega(nomega-2)*ai)**3/)
              amat(4,:)=(/dcmplx(1.0d0,0.0d0), -1.0d0/(omega(nomega-2)*ai), 1.0d0/(omega(nomega-2)*ai)**2, -1.0d0/(omega(nomega-2)*ai)**3/)

              bmat(1,1)=ff(iorb,jorb,is,nomega-1)
              bmat(2,1)=dconjg(ff(jorb,iorb,is,nomega-1))
              bmat(3,1)=ff(iorb,jorb,is,nomega-2)
              bmat(4,1)=dconjg(ff(jorb,iorb,is,nomega-2))

              call zgesv(4,1,amat,4,ipiv,bmat,4,info)

              high(iorb,jorb,is)=bmat(1,1)
              moment(iorb,jorb,is,1)=bmat(2,1)
              moment(iorb,jorb,is,2)=bmat(3,1)
              moment(iorb,jorb,is,3)=bmat(4,1)              
            enddo
          enddo
        end do
        deallocate(amat)
        deallocate(bmat)
        deallocate(ipiv)
      endif
    end if

    do is=1, ns
      high(:,:,is)=(transpose(dconjg(high(:,:,is)))+high(:,:,is))/2.0d0    
      do ii=1, 3
        moment(:,:,is,ii)=(transpose(dconjg(moment(:,:,is,ii)))+moment(:,:,is,ii))/2.0d0
      enddo
    enddo

  end subroutine FLocDyn_M


  subroutine FLatDyn_M(norb,ns,nk,nomega,omega, ff, isgreen,highzero,moment, high)
    implicit none
    integer, intent(in) :: norb, nk,ns,nomega, isgreen,highzero
    double precision, intent(in) :: omega(0:(nomega-1))
    complex*16, intent(in) :: ff(norb,norb,ns,nk,0:(nomega-1))
    complex*16,intent(out) :: moment(norb,norb,ns,nk,3), high(norb,norb,ns,nk)

    integer :: ik

    moment=0.0d0
    high=0.0d0
    do ik=1, nk
      call FLocDyn_M(norb,ns,nomega,omega,ff(:,:,:,ik,:), isgreen,highzero,moment(:,:,:,ik,:),high(:,:,:,ik))      
    enddo

  end subroutine FLatDyn_M

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine BLocDyn_M(norb,ns,nnu,nu, ff,oddzero,highzero,moment,high)
    implicit none
    integer, intent(in) :: norb, ns,nnu,oddzero,highzero
    double precision, intent(in) :: nu(0:(nnu-1))
    complex*16, intent(in) :: ff(norb,norb,ns,ns,0:(nnu-1))
    complex*16,intent(out) :: moment(norb,norb,ns,ns,3), high(norb,norb,ns,ns)

    integer :: is, js,info,iorb,jorb,ii
    complex*16 :: moment_temp(norb,norb,ns,ns,3), high_temp(norb,norb,ns,ns), ai
    integer, allocatable :: ipiv(:)
    complex*16, allocatable :: amat(:,:), bmat(:,:)

    moment=0.0d0
    high=0.0d0

    ai=dcmplx(0.0d0, 1.0d0)
    if (oddzero .eq. 1) then
      if (highzero .eq. 1) then
        moment(:,:,:,:,2)=ff(:,:,:,:,nnu-1)*(nu(nnu-1)*ai)**2
      else
        moment(:,:,:,:,2)=(ff(:,:,:,:,nnu-1)-ff(:,:,:,:,nnu-2))&
          *-1.0d0*(nu(nnu-1)*ai*nu(nnu-2)*ai)**2/(nu(nnu-1)*ai+nu(nnu-2)*ai)/(nu(nnu-1)*ai-nu(nnu-2)*ai)
        high=ff(:,:,:,:,nnu-1)-moment(:,:,:,:,2)/(nu(nnu-1)*ai)**2              
      endif

    else
      if (highzero .eq. 1) then
        do is=1, ns
          do js=1, ns
            do iorb=1, norb
              do jorb=1, norb
                moment(iorb,jorb,is,js,1) &
                  =moment(iorb,jorb,is,js,1) &
                  +(ff(iorb,jorb,is,js,nnu-1)-dconjg(ff(jorb,iorb,js,is,nnu-1))) &
                  /2.0d0*(nu(nnu-1)*ai)

                moment(iorb,jorb,is,js,2) &
                  =moment(iorb,jorb,is,js,2) &
                  +(ff(iorb,jorb,is,js,nnu-1)+dconjg(ff(jorb,iorb,js,is,nnu-1))) &
                  /2.0d0*(nu(nnu-1)*ai)**2
              enddo
            enddo
          enddo
        enddo
      else
        allocate(amat(4,4))
        allocate(bmat(4,1))
        allocate(ipiv(4))
        do is=1, ns
          do js=1, ns          
            do iorb=1, norb
              do jorb=1, norb

                amat=0.0d0
                bmat=0.0d0

                amat(1,:)=(/dcmplx(1.0d0,0.0d0), 1.0d0/(nu(nnu-1)*ai), 1.0d0/(nu(nnu-1)*ai)**2, 1.0d0/(nu(nnu-1)*ai)**3/)
                amat(2,:)=(/dcmplx(1.0d0,0.0d0), -1.0d0/(nu(nnu-1)*ai), 1.0d0/(nu(nnu-1)*ai)**2, -1.0d0/(nu(nnu-1)*ai)**3/)          
                amat(3,:)=(/dcmplx(1.0d0,0.0d0), 1.0d0/(nu(nnu-2)*ai), 1.0d0/(nu(nnu-2)*ai)**2, 1.0d0/(nu(nnu-2)*ai)**3/)
                amat(4,:)=(/dcmplx(1.0d0,0.0d0), -1.0d0/(nu(nnu-2)*ai), 1.0d0/(nu(nnu-2)*ai)**2, -1.0d0/(nu(nnu-2)*ai)**3/)

                bmat(1,1)=ff(iorb,jorb,is,js,nnu-1)
                bmat(2,1)=dconjg(ff(jorb,iorb,js,is,nnu-1))
                bmat(3,1)=ff(iorb,jorb,is,js,nnu-2)
                bmat(4,1)=dconjg(ff(jorb,iorb,js,is,nnu-2))

                call zgesv(4,1,amat,4,ipiv,bmat,4,info)

                high(iorb,jorb,is,js)=bmat(1,1)
                moment(iorb,jorb,is,js,1)=bmat(2,1)
                moment(iorb,jorb,is,js,2)=bmat(3,1)
                moment(iorb,jorb,is,js,3)=bmat(4,1)              
              enddo
            enddo
          end do
        enddo
        deallocate(amat)
        deallocate(bmat)
        deallocate(ipiv)
      endif
    endif


    moment_temp=moment
    high_temp=high
    moment=0.0d0
    high=0.0d0

    do iorb=1, norb
      do jorb=1, norb
        do is=1, ns
          do js=1, ns
            high(iorb,jorb,is,js)=(dconjg(high_temp(jorb,iorb,js,is))+high_temp(iorb,jorb,is,js))/2.0d0        
            do ii=1, 3
              moment(iorb,jorb,is,js,ii)=(dconjg(moment_temp(jorb,iorb,js,is,ii))+moment_temp(iorb,jorb,is,js,ii))/2.0d0
            enddo
          enddo
        enddo
      enddo
    enddo
  end subroutine BLocDyn_M


  subroutine BLatDyn_M(norb,ns,nk,nnu,nu, ff,oddzero,highzero,moment,high)
    implicit none
    integer, intent(in) :: norb, ns,nk,nnu,oddzero,highzero
    double precision, intent(in) :: nu(0:(nnu-1))
    complex*16, intent(in) :: ff(norb,norb,ns,ns,nk,0:(nnu-1))
    complex*16,intent(out) :: moment(norb,norb,ns,ns,nk,3), high(norb,norb,ns,ns,nk)

    integer :: ik

    moment=0.0d0
    high=0.0d0
    do ik=1, nk
      call BLocDyn_M(norb,ns,nnu,nu,ff(:,:,:,:,ik,:),oddzero,highzero,moment(:,:,:,:,ik,:),high(:,:,:,:,ik))
    enddo

  end subroutine BLatDyn_M

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  subroutine Dyn_T2F(ntau,tau,ftau,nf,freq,ff)
    implicit none
    integer, intent(in) :: ntau,nf
    double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
    complex*16, intent(in) :: ftau(0:(ntau-1))
    complex*16, intent(out) :: ff(0:(nf-1))

    integer :: if,itau
    complex*16 :: temp(0:(ntau-1)),temp2(0:(ntau-1)), ai

    ff=0.0d0
    ai=dcmplx(0.0d0, 1.0d0)
    do if=0, nf-1
      temp=0.0d0
      temp2=0.0d0
      do itau=0, ntau-1
        temp(itau)=ftau(itau)*cdexp(freq(if)*tau(itau)*ai)
      enddo
      call fderiv_dcmplx(-1,ntau,tau(0),temp(0),temp2(0))
      ff(if)=temp2(ntau-1)
    enddo
  end subroutine Dyn_T2F

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! subroutine FLocDyn_T2F_v0(norb,ns,ntau,tau,ftau,nf,freq,ff)
!   implicit none
!   integer, intent(in) :: norb,ns,ntau,nf
!   double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
!   complex*16, intent(in) :: ftau(norb,norb,ns,0:(ntau-1))
!   complex*16, intent(out) :: ff(norb,norb,ns,0:(nf-1))

!   integer :: iorb,jorb,itau,ifreq,is
!   complex*16 :: tempf(0:(nf-1))

!   ff=0.0d0

!   do iorb=1, norb
!     do jorb=1, norb
!       do is=1, ns
!         call Dyn_T2F(ntau,tau,ftau(iorb,jorb,is,:),nf,freq,tempf)
!         do ifreq=0, nf-1
!           ff(iorb,jorb,is,ifreq)=tempf(ifreq)
!         enddo
!       enddo
!     enddo
!   enddo

! end subroutine FLocDyn_T2F_V0


  subroutine FLocDyn_T2F(norb,ns,ntau,tau,ftau,nf,freq,ff)
    implicit none
    integer, intent(in) :: norb,ns,ntau,nf
    double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
    complex*16, intent(in) :: ftau(norb,norb,ns,0:(ntau-1))
    complex*16, intent(out) :: ff(norb,norb,ns,0:(nf-1))


    integer :: iorb,jorb,itau,ifreq,is,itheta,ierr

    double precision :: pi, beta, taurad_finu((-ntau):(ntau-1))
    complex*16 :: ff_finu((-2*nf+1):(2*nf-1)), ftau_finu((-ntau):(ntau-1))
    integer*8 :: ntau_finu, nf_finu    
    integer*8, allocatable :: null


    pi=datan2(1.0d0,1.0d0)*4.0d0
    beta=pi/freq(0)

    ntau_finu=2*ntau
    nf_finu=4*nf-1
    do itau=0, ntau-1
      itheta=ttind(itau, ntau)
      taurad_finu(itau)=tau(itau)/beta*pi
      taurad_finu(-itau-1)=-taurad_finu(itau)
    enddo

    ff=0.0d0

    do iorb=1, norb
      do jorb=1, norb
        do is=1, ns
          ftau_finu=0.0d0
          ff_finu=0.0d0
          do itau=0, ntau-1
            ftau_finu(itau)=ftau(iorb,jorb,is,itau)*dsqrt(tau(itau)*(beta-tau(itau)))*pi/ntau
            ftau_finu(itau-ntau)=-ftau_finu(itau)
          enddo
          call finufft1d1(ntau_finu,taurad_finu,ftau_finu,1, 1.0d-12, nf_finu,ff_finu,null,ierr)
          do ifreq=0, 2*nf-1
            if (modulo(ifreq, 2) .eq. 1) then
              ff(iorb,jorb,is,(ifreq-1)/2)=ff_finu(ifreq)/2.0d0
            endif
          enddo
        enddo
      enddo
    enddo

  end subroutine FLocDyn_T2F



  subroutine FLatDyn_T2F(norb,ns,nk,ntau,tau,ftau,nf,freq,ff)
    implicit none
    integer, intent(in) :: norb,ns,nk,ntau,nf
    double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
    complex*16, intent(in) :: ftau(norb,norb,ns,nk,0:(ntau-1))
    complex*16, intent(out) :: ff(norb,norb,ns,nk,0:(nf-1))
    integer :: ik


    ff=0.0d0
    do ik=1, nk
      call FLocDyn_T2F(norb,ns,ntau,tau,ftau(:,:,:,ik,:),nf,freq,ff(:,:,:,ik,:))
    enddo

  end subroutine FLatDyn_T2F

! subroutine FLatDyn_T2F_v0(norb,ns,nk,ntau,tau,ftau,nf,freq,ff)
!   implicit none
!   integer, intent(in) :: norb,ns,nk,ntau,nf
!   double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
!   complex*16, intent(in) :: ftau(norb,norb,ns,nk,0:(ntau-1))
!   complex*16, intent(out) :: ff(norb,norb,ns,nk,0:(nf-1))
!   integer :: ik


!   ff=0.0d0
!   do ik=1, nk
!     call FLocDyn_T2F_v0(norb,ns,ntau,tau,ftau(:,:,:,ik,:),nf,freq,ff(:,:,:,ik,:))
!   enddo

! end subroutine FLatDyn_T2F_V0

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine BLocDyn_T2F(norb,ns,ntau,tau,ftau,nf,freq,ff)
    implicit none
    integer, intent(in) :: norb,ns,ntau,nf
    double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
    complex*16, intent(in) :: ftau(norb,norb,ns,ns,0:(ntau-1))
    complex*16, intent(out) :: ff(norb,norb,ns,ns,0:(nf-1))

    integer :: iorb,jorb,itau,ifreq,is,js,itheta,ierr

    double precision :: pi, beta, taurad_finu(0:(ntau-1))
    complex*16 :: ff_finu((-nf+1):(nf-1)), ftau_finu(0:(ntau-1))
    integer*8 :: ntau_finu, nf_finu    
    integer*8, allocatable :: null


    pi=datan2(1.0d0,1.0d0)*4.0d0
    beta=2.0d0*pi/freq(1)

    ntau_finu=ntau
    nf_finu=2*nf-1
    do itau=0, ntau-1
      itheta=ttind(itau, ntau)
      taurad_finu(itau)=tau(itau)/beta*2.0d0*pi
    enddo

    ff=0.0d0

    do iorb=1, norb
      do jorb=1, norb
        do is=1, ns
          do js=1, ns          
            ftau_finu=0.0d0
            ff_finu=0.0d0
            do itau=0, ntau-1
              ftau_finu(itau)=ftau(iorb,jorb,is,js,itau)*dsqrt(tau(itau)*(beta-tau(itau)))*pi/ntau
            enddo
            call finufft1d1(ntau_finu,taurad_finu,ftau_finu,1, 1.0d-12, nf_finu,ff_finu,null,ierr)
            do ifreq=0, nf-1
              ff(iorb,jorb,is,js,ifreq)=ff_finu(ifreq)
            enddo
          enddo
        enddo
      enddo
    enddo

  end subroutine BLocDyn_T2F


! subroutine BLocDyn_T2F_v0(norb,ns,ntau,tau,ftau,nf,freq,ff)
!   implicit none
!   integer, intent(in) :: norb,ns,ntau,nf
!   double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
!   complex*16, intent(in) :: ftau(norb,norb,ns,ns,0:(ntau-1))
!   complex*16, intent(out) :: ff(norb,norb,ns,ns,0:(nf-1))

!   integer :: iorb,jorb,itau,ifreq,is,js
!   complex*16 :: tempf(0:(nf-1))

!   ff=0.0d0

!   do iorb=1, norb
!     do jorb=1, norb
!       do is=1, ns
!         do js=1, ns        

!           call Dyn_T2F(ntau,tau,ftau(iorb,jorb,is,js,:),nf,freq,tempf)

!           do ifreq=0, nf-1
!             ff(iorb,jorb,is,js,ifreq)=tempf(ifreq)
!           enddo
!         enddo
!       enddo
!     enddo
!   enddo

! end subroutine BLocDyn_T2F_V0


  subroutine BLatDyn_T2F(norb,ns,nk,ntau,tau,ftau,nf,freq,ff)
    implicit none
    integer, intent(in) :: norb,ns,nk,ntau,nf
    double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
    complex*16, intent(in) :: ftau(norb,norb,ns,ns,nk,0:(ntau-1))
    complex*16, intent(out) :: ff(norb,norb,ns,ns,nk,0:(nf-1))


    integer :: ik

    ff=0.0d0
    do ik=1, nk
      call BLocDyn_T2F(norb,ns,ntau,tau,ftau(:,:,:,:,ik,:),nf,freq,ff(:,:,:,:,ik,:))
    enddo

  end subroutine BLatDyn_T2F


! subroutine BLatDyn_T2F_v0(norb,ns,nk,ntau,tau,ftau,nf,freq,ff)
!   implicit none
!   integer, intent(in) :: norb,ns,nk,ntau,nf
!   double precision, intent(in) :: tau(0:(ntau-1)), freq(0:(nf-1))
!   complex*16, intent(in) :: ftau(norb,norb,ns,ns,nk,0:(ntau-1))
!   complex*16, intent(out) :: ff(norb,norb,ns,ns,nk,0:(nf-1))


!   integer :: ik

!   ff=0.0d0
!   do ik=1, nk
!     call BLocDyn_T2F_v0(norb,ns,ntau,tau,ftau(:,:,:,:,ik,:),nf,freq,ff(:,:,:,:,ik,:))
!   enddo

! end subroutine BLatDyn_T2F_v0  

!!!!!!!!!!!!!!!!!!!!!!!!!!



end Module Fourier


