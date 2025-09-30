Module Embedding
  implicit None

  private
! public :: Embedding
  public :: &
! interface Embedding
!    module procedure &
    FLocStc, &
    FLocDyn, &
    FLatStc, &
    FLatDyn, &
    BLocStc, &
    BLocDyn, &
    BLatStc, &
    BLatDyn
! end interface Embedding

contains

  subroutine FLocStc(norb,ns,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb,ns,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns)
    complex*16,intent(out) :: ff(norb,norb,ns)

    integer :: is
    complex*16 :: tempmat(norb,norbc)

    ff=0.0d0
    do is=1, ns      
      tempmat=0.0d0
      call zgemm('n','n',norb,norbc,norbc,(1.0d0,0.0d0),projector(1,1,is),norb,ffc(1,1,is),norbc,(0.0d0,0.0d0),tempmat,norb)
      call zgemm('n','c',norb,norb,norbc,(1.0d0,0.0d0),tempmat,norb,projector(1,1,is),norb,(0.0d0,0.0d0),ff(1,1,is),norb)
    enddo

  end subroutine FLocStc

  subroutine FLatStc(norb,ns,nk,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb, nk,ns,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns)
    complex*16,intent(out) :: ff(norb,norb,ns,nk)

    integer :: ik

    ff=0.0d0
    do ik=1, nk
      call FLocStc(norb,ns,ffc,norbc,projector,ff(:,:,:,ik))
    enddo

  end subroutine FLatStc


  subroutine FLocDyn(norb,ns,nf,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb, ns,norbc,nf
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns,0:(nf-1))
    complex*16,intent(out) :: ff(norb,norb,ns,0:(nf-1))

    integer :: ifreq

    ff=0.0d0
    do ifreq=0, nf-1
      call FLocStc(norb,ns,ffc(:,:,:,ifreq),norbc,projector,ff(:,:,:,ifreq))
    enddo

  end subroutine FLocDyn

  subroutine FLatDyn(norb,ns,nk,nf,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb, nk,nf,ns,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns,0:(nf-1))
    complex*16,intent(out) :: ff(norb,norb,ns,nk,0:(nf-1))

    integer :: ifreq

    ff=0.0d0
    do ifreq=0, nf-1
      call FLatStc(norb,ns,nk,ffc(:,:,:,ifreq),norbc,projector,ff(:,:,:,:,ifreq))
    enddo

  end subroutine FLatDyn


  subroutine BLocStc(norb,ns,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb,ns,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns,ns)
    complex*16,intent(out) :: ff(norb,norb,ns,ns)

    integer :: is,js
    complex*16 :: tempmat(norb,norbc)

    ff=0.0d0
    do is=1, ns
      do js=1, ns
        tempmat=0.0d0
        call zgemm('n','n',norb,norbc,norbc,(1.0d0,0.0d0),projector(1,1,is),norb,ffc(1,1,is,js),norbc,(0.0d0,0.0d0),tempmat,norb)
        call zgemm('n','c',norb,norb,norbc,(1.0d0,0.0d0),tempmat,norb,projector(1,1,js),norb,(0.0d0,0.0d0),ff(1,1,is,js),norb)
      enddo
    enddo

  end subroutine BLocStc


  subroutine BLatStc(norb,ns,nk,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb,ns,nk,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns,ns)
    complex*16,intent(out) :: ff(norb,norb,ns,ns,nk)

    integer :: ik

    do ik=1, nk
      call BLocStc(norb,ns,ffc,norbc,projector,ff(:,:,:,:,ik))
    enddo

  end subroutine BLatStc


  subroutine BLocDyn(norb,ns,nf,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb,ns,nf,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns,ns,0:(nf-1))
    complex*16,intent(out) :: ff(norb,norb,ns,ns,0:(nf-1))

    integer :: ifreq

    do ifreq=0, nf-1
      call BLocStc(norb,ns,ffc(:,:,:,:,ifreq),norbc,projector,ff(:,:,:,:,ifreq))
    enddo

  end subroutine BLocDyn


  subroutine BLatDyn(norb,ns,nk,nf,ffc,norbc,projector,ff)
    implicit none
    integer, intent(in) :: norb,ns,nk,nf,norbc
    complex*16, intent(in) :: projector(norb,norbc,ns), ffc(norbc,norbc,ns,ns,0:(nf-1))
    complex*16,intent(out) :: ff(norb,norb,ns,ns,nk,0:(nf-1))

    integer :: ifreq

    do ifreq=0, nf-1
      call BLatStc(norb,ns,nk,ffc(:,:,:,:,ifreq),norbc,projector,ff(:,:,:,:,:,ifreq))
    enddo

  end subroutine BLatDyn


end Module Embedding
