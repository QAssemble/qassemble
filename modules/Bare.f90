Module Bare
  use Common
  implicit None

  private
! public: G0

! interface G0
!    module procedure &
  public :: &
    FLatDyn, &
    FLocDyn, &
    FTau, &
    FFreq, &    
    FLocTau, &
    FLatTau, &    

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
      call FLocDyn(norb,ns,nfreq,hlatt(:,:,:,ik), freq,glatt(:,:,:,ik,:))
    enddo

  end subroutine FLatFreq

  subroutine FLocFreq(norb,ns,nfreq,hloc,freq,gloc)
    implicit none

    integer, intent(in) :: norb,ns,nfreq
    double precision, intent(in) :: freq(0:(nfreq-1))
    complex*16, intent(in) :: hloc(norb,norb,ns)
    complex*16,intent(out) :: gloc(norb,norb,ns,0:(nfreq-1))

    integer :: is,ifreq,iorb
    double precision :: w(norb)
    complex*16 :: tempmat(norb,norb), ai, ffreq(0:(nfreq-1),norb)

    ai=dcmplx(0.0d0, 1.0d0)
    gloc=0.0d0

    do is=1, ns
      tempmat=hloc(:,:,is)
      call hermitianeigen_dcmplx(norb, w, tempmat)

      ffreq=0.0d0
      do iorb=1, norb
        call FFreq(nfreq, freq, w(iorb), ffreq(:,iorb))
      enddo

      do ifreq=0, nfreq-1
        do iorb=1, norb
          do jorb=1, norb
            tempmat2(iorb,jorb)=tempmat(iorb, jorb)*ffreq(ifreq, iorb)
          enddo
        enddo

        call zgemm('n','c',norb,norb,norb,(1.0d0,0.0d0),tempmat2,norb,tempmat,norb,(0.0d0,0.0d0),gloc(1,1,is,ifreq),norb)
      enddo
    enddo

  end subroutine FLocFreq


  subroutine FTau(ntau, tau, energy, ftau)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: ntau
    double precision, intent(in) :: tau(0:(ntau-1)), energy
    complex*16, intent(out) :: ftau(0:(ntau-1))    
    
    integer :: unitnum
    double precision :: machep,taumod,taunew, beta

    integer :: itau
    
    
    beta= tau(0)*2.0d0/(dcos(pi*(ntau-0.5d0)/dble(ntau))+1.0d0)    
    
    machep = epsilon ( machep )
    
    do itau=0, ntau-1
      taumod=modulo(tau(itau), beta)
      unitnum=nint(tau(itau)-taumod)/beta
      if (taumod .lt. machep) then
        unitnum=unitnum-1
      endif
      taunew=tau(itau)-beta*unitnum          
      
      
      if (energy .gt. 0) then
        ftau(itau)=(-1)**(unitnum+1)*dexp(-energy*taunew)*(1-1.0d0/(dexp(energy*beta)+1))
      else
        ftau(itau)=(-1)**(unitnum+1)*dexp(-energy*(taunew-beta))*(1.0d0/(dexp(energy*beta)+1))
      endif
    enddo
  end subroutine FTau


  subroutine BTau(ntau, tau, energy, btau)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: ntau
    double precision, intent(in) :: tau(0:(ntau-1)), energy
    complex*16, intent(out) :: btau(0:(ntau-1))    
    
    integer :: unitnum
    double precision :: machep,taumod,taunew, beta
    
    integer :: itau
    
    
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
        btau(itau)=-dexp(-energy*taunew)*(1-1.0d0/(dexp(energy*beta)-1))
      else
        btau(itau)=-dexp(-energy*(taunew-beta))*(1.0d0/(dexp(energy*beta)-1))
      endif
    enddo
  end subroutine BTau
  
  subroutine FFreq(nfreq, freq, energy, ffreq)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: nfreq
    double precision, intent(in) :: freq(0:(nfreq-1)), energy
    complex*16, intent(out) :: ffreq(0:(nfreq-1))
    
    integer :: ifreq
    complex*16 :: ai
    
    ai=dcmplx(0.0d0, 1.0d0)    
    
    do ifreq=0, nfreq-1
      ffreq(ifreq)=1.0d0/(ai*freq(ifreq)-energy)
    enddo
  end subroutine FFreq


  subroutine BFreq(nfreq, freq, energy, bfreq)
! from wikipedia: https://en.wikipedia.org/wiki/Matsubara_frequency
    implicit none
    
    integer, intent(in) :: nfreq
    double precision, intent(in) :: freq(0:(nfreq-1)), energy
    complex*16, intent(out) :: bfreq(0:(nfreq-1))
    
    
    integer :: ifreq
    complex*16 :: ai
    
    ai=dcmplx(0.0d0, 1.0d0)    
    
    do ifreq=0, nfreq-1
      bfreq(ifreq)=1.0d0/(ai*freq(ifreq)-energy)
    enddo
  end subroutine BFreq  
  
end Module Bare
