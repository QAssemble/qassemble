Module Projection
  implicit None

  private
! public :: Projection
  public :: &
! interface Projection
!    module procedure &
    FLocStc, &
    FLatStc, &
    FLocDyn, &  
    FLatDyn, &
    BLocStc, &
    BLatStc, &
    BLocDyn, &  
    BLatDyn    


! end interface Projection  


contains

  subroutine FLocStc(norb,ns,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, ns,norbc
    complex*16, intent(in) :: ff(norb,norb,ns), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns)
    
    integer :: is
    complex*16 :: tempmat(norb,norbc)

    ffc=0.0d0
    do is=1, ns
      tempmat=0.0d0
      call zgemm('n','n',norb,norbc,norb,(1.0d0,0.0d0),ff(1,1,is),norb,projector(1,1,is),norb,(0.0d0,0.0d0),tempmat,norb)
      call zgemm('c','n',norbc,norbc,norb,(1.0d0,0.0d0),projector(1,1,is),norb,tempmat,norb,(0.0d0,0.0d0),ffc(1,1,is),norbc)
    enddo

  end subroutine FLocStc

  
  subroutine FLatStc(norb,ns,nk,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, nk,ns,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,nk), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns)
    complex*16 :: tempmat(norbc,norbc,ns)

    integer :: ik

    ffc=0.0d0
    do ik=1, nk
      tempmat = 0.0d0
      call FLocStc(norb,ns,ff(:,:,:,ik),norbc,projector,tempmat)
      ffc = ffc + tempmat/nk
    enddo
    
    
  end subroutine FLatStc

  
  subroutine FLocDyn(norb,ns,nf,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, nf,ns,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,0:(nf-1)), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns,0:(nf-1))

    integer :: ifreq

    ffc=0.0d0
    do ifreq=0, nf-1
      call FLocStc(norb,ns,ff(:,:,:,ifreq),norbc,projector,ffc(:,:,:,ifreq))
    enddo

  end subroutine FLocDyn
  
  subroutine FLatDyn(norb,ns,nk,nf,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, nk,ns,nf,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,nk,0:(nf-1)), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns,0:(nf-1))

    integer :: ifreq
    
    do ifreq=0, nf-1
      call FLatStc(norb,ns,nk,ff(:,:,:,:,ifreq),norbc,projector,ffc(:,:,:,ifreq))      
    enddo

  end subroutine FLatDyn

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  subroutine BLocStc(norb,ns,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, ns,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,ns), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns,ns)

    integer :: is,js
    complex*16 :: tempmat(norb,norbc)

    ffc=0.0d0
    do is=1, ns
      do js=1, ns        
        tempmat=0.0d0
        call zgemm('n','n',norb,norbc,norb,(1.0d0,0.0d0),ff(1,1,is,js),norb,projector(1,1,js),norb,(0.0d0,0.0d0),tempmat,norb)
        call zgemm('c','n',norbc,norbc,norb,(1.0d0,0.0d0),projector(1,1,is),norb,tempmat,norb,(0.0d0,0.0d0),ffc(1,1,is,js),norbc)
      enddo
    enddo
    
  end subroutine BLocStc

  subroutine BLatStc(norb,ns,nk,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, nk,ns,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,ns,nk), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns,ns)

    integer :: ik
    complex*16 :: tempmat(norbc,norbc,ns,ns)

    ffc=0.0d0
    do ik=1, nk
      tempmat = 0.0d0
      call BLocStc(norb,ns,ff(:,:,:,:,ik),norbc,projector,tempmat)
      ffc = ffc + tempmat/nk
    enddo

  end subroutine BLatStc


  subroutine BLocDyn(norb,ns,nf,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, ns,nf,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,ns,0:(nf-1)), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns,ns,0:(nf-1))

    integer :: ifreq

    ffc=0.0d0
    do ifreq=0, nf-1
      call BLocStc(norb,ns,ff(:,:,:,:,ifreq),norbc,projector,ffc(:,:,:,:,ifreq))
    enddo

  end subroutine BLocDyn
  


  subroutine BLatDyn(norb,ns,nk,nf,ff,norbc,projector,ffc)
    implicit none
    integer, intent(in) :: norb, nk,ns,nf,norbc
    complex*16, intent(in) :: ff(norb,norb,ns,ns,nk,0:(nf-1)), projector(norb,norbc,ns)
    complex*16,intent(out) :: ffc(norbc,norbc,ns,ns,0:(nf-1))

    integer :: ifreq


    ffc=0.0d0
    do ifreq=0, nf-1
      call BLatStc(norb,ns,nk,ff(:,:,:,:,:,ifreq),norbc,projector,ffc(:,:,:,:,ifreq))
    enddo

  end subroutine BLatDyn



end Module Projection
