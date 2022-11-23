program TestMoment
  use Fourier
  use Common
  implicit none

  integer :: iomega, ik, is1,is2,iorb1,iorb2, ii,ntau,itau,nomega,itheta,norb,ns,nk, is,iorb,jorb

  double precision :: beta, pi
  complex*16 :: err, errmax, ai

  double precision, allocatable :: omega(:), tau(:), eig(:)
  complex*16, allocatable :: &
    tempmat(:,:),tempmat1(:,:),tempmat2(:,:),eigmat(:,:),&
    hmat(:,:,:,:), identity(:,:),glatt0(:,:,:,:,:), flatdyn_t_ref(:,:,:,:,:), glatt02(:,:,:,:,:),&
    flatdyn_t(:,:,:,:,:),flatdyn_t2(:,:,:,:,:),flatdyn_moment(:,:,:,:,:),flatdyn_high(:,:,:,:)


  nomega=3000
  ntau=nomega
  nc=ntau

  ! allocate(tempmat(norb,norb))
  ! allocate(tempmat1(norb,norb))
  ! allocate(tempmat2(norb,norb))
  allocate(eig(norb))
  allocate(eigmat(norb,norb))
  allocate(hmat(norb,norb,ns,nk))
  allocate(identity(norb,norb))
  allocate(glatt0(norb,norb,ns,nk,0:(nomega-1)))
  allocate(glatt02(norb,norb,ns,nk,0:(nomega-1)))  
  allocate(flatdyn_t_ref(norb,norb,ns,nk,0:(ntau-1)))
  allocate(flatdyn_t(norb,norb,ns,nk,0:(ntau-1)))
  allocate(flatdyn_t2(norb,norb,ns,nk,0:(ntau-1)))  
  allocate(flatdyn_moment(norb,norb,ns,nk,3))
  allocate(flatdyn_high(norb,norb,ns,nk))

  allocate(omega(0:(nomega-1)))
  omega=0.0d0


  allocate(tau(0:(ntau-1)))
  tau=0.0d0


  beta=1.0d0/(8.617333262145d-5*300.0d0)
  pi=datan2(1.0d0,1.0d0)*4.0d0

  omega=0.0d0

  do iorb=1, norb
    eig(iorb)=iorb*2.0d0
  enddo

  allocate(gomega(norb,norb,ns,nomega))
  gomega=0.0d0
  
  do iomega=0, nomega-1
    do is=1, ns
      do iorb=1, norb
        gomega(iorb,iorb,is,iomega)=1.0d0/(omega(iomega)*ai-eig(iorb))
      enddo
    enddo
    gomega2(1,1,1,iomega)=gomega(1,1,1,iomega)*gomega(2,2,1,iomega)    
  enddo


  call FLocDyn_F2T(norb,ns,nomega,omega,gomega,moment,ntau,tau,ftau)



  

  call FLocDyn_Cheby_Coeff(norb,ns,ntau,tau,ftau, nc,chebyshev_coeff)  

  errmax=0.0d0
  do itau=0, ntau-1
    do is=1, ns
      do iorb1=1, norb
        do iorb2=1, norb
          ! err=flatdyn_t(iorb1,iorb2,is,1,itau)-flatdyn_t_ref(iorb1,iorb2,is,1,itau)
          err=flatdyn_t(iorb1,iorb2,is,1,itau)-flatdyn_t2(iorb1,iorb2,is,1,itau)          
          if (cdabs(err) .gt. cdabs(errmax)) then
            errmax=err
          end if
        enddo
      end do
    end do
  enddo
  print *, 'Flocdyn_F2T', errmax

! open(unit=5, file='ftau.dat')
! do itau=0, ntau-1
!   do is=1, ns
!     do iorb1=1, norb
!       do iorb2=1, norb
!         write(5, '(7(2x,f12.6))') tau(itau),flatdyn_t(iorb1,iorb2,is,1,itau), flatdyn_t_ref(iorb1,iorb2,is,1,itau), flatdyn_t(iorb1,iorb2,is,1,itau)-flatdyn_t2(iorb1,iorb2,is,1,itau)
!       end do
!     enddo
!   end do
! end do
! close(5)


  call FLatDyn_M(norb,ns,nk,nomega,omega,glatt0,1,1,flatdyn_moment,flatdyn_high) 

  call FLatDyn_F2T(norb,ns,nk,nomega,omega,glatt0,flatdyn_moment,ntau,tau,flatdyn_t)
  call FLatDyn_F2T_v0(norb,ns,nk,nomega,omega,glatt0,flatdyn_moment,ntau,tau,flatdyn_t2)  
  ! call FLatDyn_Normalization(norb,ns,nk,ntau,tau,flatdyn_t)


  errmax=0.0d0
  do ik=1, nk
    do itau=0, ntau-1
      do is=1, ns
        do iorb1=1, norb
          do iorb2=1, norb
            ! err=flatdyn_t(iorb1,iorb2,is,ik,itau)-flatdyn_t_ref(iorb1,iorb2,is,ik,itau)
            err=flatdyn_t(iorb1,iorb2,is,ik,itau)-flatdyn_t2(iorb1,iorb2,is,ik,itau)            
            if (cdabs(err) .gt. cdabs(errmax)) then
              errmax=err
            end if
          enddo
        end do
      end do
    enddo
  enddo
  print *, 'FLatdyn_F2T', errmax  

! open(unit=5, file='ftau.dat')
! do itau=0, ntau-1
!   do ik=1, nk
!     do is=1, ns
!       do iorb1=1, norb
!         do iorb2=1, norb
!           write(5, '(7(2x,f12.6))') tau(itau),flatdyn_t(iorb1,iorb2,is,ik,itau), flatdyn_t_ref(iorb1,iorb2,is,ik,itau), flatdyn_t(iorb1,iorb2,is,ik,itau)-flatdyn_t_ref(iorb1,iorb2,is,ik,itau)
!         end do
!       enddo
!     enddo
!   end do
! end do
! close(5)  


end program TestMoment
