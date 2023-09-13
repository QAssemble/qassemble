Module Bare
  use Common
  implicit None

  private
! public: G0

! interface G0
!    module procedure &
  public :: &
    FLatFreq, &
    FLocFreq, &
    FTau, &
    BTau, &
    FFreq, &
    BLocFreq, &
    BLatFreq, &
    FLocTau, &
    FLatTau, &
    BLocTau, &
    BLatTau, &
    BFreq

! end interface Dyson

contains

  subroutine FLatFreq(norb,ns,nk,nfreq,hlatt,freq,glatt)
    implicit none

    integer, intent(in) :: norb,ns,nk,nfreq
    double precision, intent(in) :: freq(0:(nfreq-1))
    complex*16, intent(in) :: hlatt(norb,norb,ns,nk)
    complex*16,intent(out) :: glatt(norb,norb,ns,nk,0:(nfreq-1))

    integer :: ik

    glatt=0.0d0

    do ik=1, nk
      call FLocFreq(norb,ns,nfreq,hlatt(:,:,:,ik), freq,glatt(:,:,:,ik,:))
    enddo

  end subroutine FLatFreq

  subroutine FLocFreq(norb,ns,nfreq,hloc,freq,gloc)
    implicit none

    integer, intent(in) :: norb,ns,nfreq
    double precision, intent(in) :: freq(0:(nfreq-1))
    complex*16, intent(in) :: hloc(norb,norb,ns)
    complex*16,intent(out) :: gloc(norb,norb,ns,0:(nfreq-1))

    integer :: is,ifreq,iorb,jorb
    double precision :: w(norb)
    complex*16 :: tempmat(norb,norb), ai, gfreq(0:(nfreq-1),norb),tempmat2(norb,norb)

    ai=dcmplx(0.0d0, 1.0d0)
    gloc=0.0d0

    do is=1, ns
      tempmat=hloc(:,:,is)
      call hermitianeigen_dcmplx(norb, w, tempmat)

      gfreq=0.0d0
      do iorb=1, norb
        call FFreq(nfreq, freq, w(iorb), gfreq(:,iorb))
      enddo

      do ifreq=0, nfreq-1
        do iorb=1, norb
          do jorb=1, norb
            tempmat2(iorb,jorb)=tempmat(iorb, jorb)*gfreq(ifreq, jorb)
          enddo
        enddo

        call zgemm('n','c',norb,norb,norb,(1.0d0,0.0d0),tempmat2,norb,tempmat,norb,(0.0d0,0.0d0),gloc(1,1,is,ifreq),norb)
      enddo
    enddo

  end subroutine FLocFreq


  subroutine FTau(ntau, tau, energy, gtau)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: ntau
    double precision, intent(in) :: tau(0:(ntau-1)), energy
    complex*16, intent(out) :: gtau(0:(ntau-1))    
    
    integer :: unitnum
    double precision :: machep,taumod,taunew, beta, pi
    integer :: itau
    
    pi = datan2(1.0d0, 1.0d0)*4.0d0
    
    beta= tau(0)/(dcos(pi*(ntau-0.5d0)/dble(ntau))+1.0d0)*2.0d0   
    machep = epsilon ( machep )

    do itau=0, ntau-1
      taumod=modulo(tau(itau), beta)
      unitnum=nint(tau(itau)-taumod)/beta
      if (taumod .lt. machep) then
        unitnum=unitnum-1
      endif
      taunew=tau(itau)-beta*unitnum          
 
      
      if (energy .gt. 0) then
        gtau(itau)=(-1)**(unitnum+1)*dexp(-energy*taunew)*(1-1.0d0/(dexp(energy*beta)+1))
      else
        gtau(itau)=(-1)**(unitnum+1)*dexp(-energy*(taunew-beta))*(1.0d0/(dexp(energy*beta)+1))
      endif
    enddo
  end subroutine FTau


  subroutine BTau(ntau, tau, energy, wtau)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: ntau
    double precision, intent(in) :: tau(0:(ntau-1)), energy
    complex*16, intent(out) :: wtau(0:(ntau-1))    
    
    integer :: unitnum
    double precision :: machep,taumod,taunew, beta, pi
    
    integer :: itau
    
    pi = datan2(1.0d0, 1.0d0)*4.0d0
    
    beta= tau(0)*2.0d0/(dcos(pi*(ntau-0.5d0)/dble(ntau))+1.0d0)    
    
    machep = epsilon ( machep )
    if (dabs(energy) .lt. 1.0d-12) then
      print *, 'zero energy in Bare.BTau. impossible'
      stop
    endif
    
    do itau=0, ntau-1
      taumod=modulo(tau(itau), beta)
      unitnum=nint(tau(itau)-taumod)/beta
      if (taumod .lt. machep) then
        unitnum=unitnum-1
      endif
      taunew=tau(itau)-beta*unitnum          
      
      if (energy .gt. 0) then
        wtau(itau)=-dexp(-energy*taunew)*(1-1.0d0/(dexp(energy*beta)-1))
      else
        wtau(itau)=-dexp(-energy*(taunew-beta))*(1.0d0/(dexp(energy*beta)-1))
      endif
    enddo
  end subroutine BTau
  
  subroutine FFreq(nfreq, freq, energy, gfreq)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: nfreq
    double precision, intent(in) :: freq(0:(nfreq-1)), energy
    complex*16, intent(out) :: gfreq(0:(nfreq-1))
    
    integer :: ifreq
    complex*16 :: ai
    
    ai=dcmplx(0.0d0, 1.0d0)    
    
    do ifreq=0, nfreq-1
      gfreq(ifreq)=1.0d0/(ai*freq(ifreq)-energy)
    enddo
  end subroutine FFreq


  subroutine BFreq(nfreq, freq, energy, wfreq)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: nfreq
    double precision, intent(in) :: freq(0:(nfreq-1)), energy
    complex*16, intent(out) :: wfreq(0:(nfreq-1))
    
    
    integer :: ifreq
    complex*16 :: ai
    
    ai=dcmplx(0.0d0, 1.0d0)    
    
    do ifreq=0, nfreq-1
      wfreq(ifreq)=1.0d0/(ai*freq(ifreq)-energy)
    enddo
  end subroutine BFreq  

  subroutine BLocFreq(norb,ns,nfreq,hloc,freq,wloc)
    implicit none

    integer, intent(in) :: norb,ns,nfreq
    double precision, intent(in) :: freq(0:(nfreq-1))
    complex*16, intent(in) :: hloc(norb,norb,ns,ns)
    complex*16,intent(out) :: wloc(norb,norb,ns,ns,0:(nfreq-1))

    integer :: is,js,ifreq,iorb,jorb
    double precision :: w(norb)
    complex*16 :: tempmat(norb,norb), ai, wfreq(0:(nfreq-1),norb),tempmat2(norb,norb)

    ai=dcmplx(0.0d0, 1.0d0)
    wloc=0.0d0

    do is=1, ns
        do js = 1, ns
           tempmat=hloc(:,:,is,js)
           call hermitianeigen_dcmplx(norb, w, tempmat)

           wfreq=0.0d0
           do iorb=1, norb
              call BFreq(nfreq, freq, w(iorb), wfreq(:,iorb))
           enddo

           do ifreq=0, nfreq-1
              do iorb=1, norb
                 do jorb=1, norb
                    tempmat2(iorb,jorb)=tempmat(iorb, jorb)*wfreq(ifreq, jorb)
                  enddo
              enddo

              call zgemm('n','c',norb,norb,norb,(1.0d0,0.0d0),tempmat2,norb,tempmat,norb,(0.0d0,0.0d0),wloc(1,1,is,js,ifreq),norb)

           enddo
        enddo
    enddo
  end subroutine BLocFreq
  
  subroutine BLatFreq(norb,ns,nk,nfreq,hlatt,freq,wlatt)
    implicit none

    integer, intent(in) :: norb, ns, nk, nfreq
    double precision, intent(in) :: freq(0:(nfreq-1))
    complex*16, intent(in) :: hlatt(norb, norb, ns, ns, nk)
    complex*16, intent(out) :: wlatt(norb, norb, ns, ns, nk, 0:(nfreq-1))

    integer :: ik

    wlatt = 0.0d0

    do ik = 1, nk
      call BLocFreq(norb, ns, nfreq, hlatt(:, :, :, :, ik), freq, wlatt(:, :, :, :, ik,:))
    enddo

  end subroutine BLatFreq

  subroutine FLocTau(norb,ns,ntau,hloc,tau,gloc)
    implicit none

    integer, intent(in) :: norb, ns, ntau
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: hloc(norb, norb, ns)
    complex*16, intent(out) :: gloc(norb, norb, ns, 0:(ntau-1))

    integer :: is, itau, iorb, jorb
    double precision :: w(norb)
    complex*16 :: tempmat(norb, norb), ai, gtau(0:(ntau-1),norb), tempmat2(norb,norb)

    ai = dcmplx(0.0d0, 1.0d0)
    gloc = 0.0d0

    do is = 1, ns
      tempmat = hloc(:,:,is)
      call hermitianeigen_dcmplx(norb, w, tempmat)

      gtau = 0.0d0
      do iorb = 1, norb
        call FTau(ntau, tau, w(iorb), gtau(:,iorb))
      enddo

      do itau = 0, ntau-1
        do iorb = 1, norb
          do jorb = 1, norb
            tempmat2(iorb, jorb) = tempmat(iorb, jorb)*gtau(itau, jorb)
          enddo
        enddo

        call zgemm('n', 'c', norb, norb, norb, (1.0d0, 0.0d0), tempmat2, norb,tempmat, norb, (0.0d0, 0.0d0), gloc(1,1,is,itau),norb)
        
      enddo
    enddo
  end subroutine FLocTau 


  subroutine FLatTau(norb,ns,nk,ntau,hlatt,tau,glatt)
    implicit none

    integer, intent(in) :: norb,ns,nk,ntau
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: hlatt(norb,norb,ns,nk)
    complex*16,intent(out) :: glatt(norb,norb,ns,nk,0:(ntau-1))

    integer :: ik

    glatt=0.0d0

    do ik=1, nk
      call FLocTau(norb,ns,ntau,hlatt(:,:,:,ik), tau,glatt(:,:,:,ik,:))
    enddo

  end subroutine FLatTau 

  subroutine BLocTau(norb, ns, ntau, hloc, tau, wloc)
    implicit none
    
    integer, intent(in) :: norb, ns, ntau
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: hloc(norb, norb, ns, ns)
    complex*16, intent(out) :: wloc(norb, norb, ns, ns,0:(ntau-1))
  
    integer :: is, js, itau, iorb, jorb
    double precision :: w(norb)
    complex*16 :: tempmat(norb, norb), ai, wtau(0:(ntau-1), norb), tempmat2(norb,norb)
  
    ai = dcmplx(0.0d0, 1.0d0)
    wloc = 0.0d0
  
    do is = 1, ns
      do js = 1, ns
        tempmat = hloc(:, :, is, js)
        call hermitianeigen_dcmplx(norb, w, tempmat)
  
        wtau = 0.0d0
        do iorb = 1, norb
          call BTau(ntau, tau, w(iorb), wtau(:,iorb))
        enddo
  
        do itau = 0, ntau-1
          do iorb = 1,norb
            do jorb = 1, norb
              tempmat2(iorb, jorb) = tempmat(iorb, jorb)*wtau(itau, jorb)
            enddo
          enddo
  
          call zgemm('n', 'c', norb, norb, norb, (1.0d0, 0.0d0), tempmat2, norb,tempmat, norb, (0.0d0,0.0d0), wloc(1,1, is, js, itau), norb)
  
        enddo
      enddo
    enddo
  
  end subroutine BLocTau

  subroutine BLatTau(norb, ns, nk, ntau, hlatt, tau, wlatt)
    implicit none

    integer, intent(in) :: norb, ns, nk, ntau
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: hlatt(norb, norb, ns, ns, nk)
    complex*16, intent(out) :: wlatt(norb, norb, ns, ns, nk, 0:(ntau-1))

    integer :: ik

    wlatt = 0.0d0

    do ik = 1, nk
      call BLocTau(norb, ns, ntau, hlatt(:,:,:,:,ik), tau, wlatt(:,:,:,:,ik,:))
    enddo

  end subroutine BLatTau

 
end Module Bare
