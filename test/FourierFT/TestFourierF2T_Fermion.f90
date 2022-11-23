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
    hmat(:,:,:,:), identity(:,:),glatt0(:,:,:,:,:), flatdyn_t_ref(:,:,:,:,:), &
    flatdyn_t(:,:,:,:,:),flatdyn_t2(:,:,:,:,:),flatdyn_moment(:,:,:,:,:),flatdyn_high(:,:,:,:)


  nomega=300
  ntau=nomega
  norb=3
  ns=2
  nk=5
  ai=dcmplx(0.0d0, 1.0d0)

  allocate(tempmat(norb,norb))
  allocate(tempmat1(norb,norb))
  allocate(tempmat2(norb,norb))
  allocate(eig(norb))
  allocate(eigmat(norb,norb))
  allocate(hmat(norb,norb,ns,nk))
  allocate(identity(norb,norb))
  allocate(glatt0(norb,norb,ns,nk,0:(nomega-1)))
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

  do iomega=0, nomega-1
    omega(iomega)=pi/beta*(2*iomega+1)
  enddo


  do itau=0, ntau-1
    itheta=ttind(itau,ntau)
    tau(itau)=beta/2.0d0*(dcos(pi*(itheta+0.5d0)/dble(ntau))+1)    
  enddo


  identity=0.0d0
  do ii=1, norb
    identity(ii,ii)=1.0d0
  enddo

! !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  do ik=1, nk
    do is=1, ns
      do iorb1=1, norb
        do iorb2=1, norb
          hmat(iorb1,iorb2,is,ik)=(iorb1+iorb2)*0.5d0+is*0.3d0*ik
          if (iorb1 .eq. iorb2) then
            hmat(iorb1,iorb2,is,ik)=hmat(iorb1,iorb2,is,ik)-6.0d0
          endif
        enddo
      enddo
    enddo
  enddo

  do ik=1, nk 
    do is=1, ns
      do iomega=0, nomega-1
        tempmat1=identity*omega(iomega)*ai-hmat(:,:,is,ik)
        call dcmplx_matinv(tempmat1, tempmat2,norb,norb)
        glatt0(:,:,is,ik,iomega)=tempmat2
      enddo
    enddo
  enddo



  do ik=1, nk  
    do is=1, ns
      tempmat=hmat(:,:,is,ik)
      call hermitianeigen_dcmplx(norb,eig,tempmat)
      print *, ik, is, eig(1), eig(norb)
      do itau=0, ntau-1        
        eigmat=0.0d0
        do iorb1=1, norb
          if (eig(iorb1) .gt. 0.0d0) then
            eigmat(iorb1,iorb1)=-dexp(-eig(iorb1)*tau(itau))*(1-1.0d0/(dexp(beta*eig(iorb1))+1.0d0))
          else
            eigmat(iorb1,iorb1)=-dexp(eig(iorb1)*(beta-tau(itau)))*(1.0d0/(dexp(beta*eig(iorb1))+1.0d0))
          endif
        enddo
        flatdyn_t_ref(:,:,is,ik,itau)=matmul(matmul(tempmat, eigmat), transpose(dconjg(tempmat)))
      enddo
    enddo
  enddo


  call FLocDyn_M(norb,ns,nomega,omega,glatt0(:,:,:,1,:),1,1,flatdyn_moment(:,:,:,1,:),flatdyn_high(:,:,:,1))  

  call FLocDyn_F2T(norb,ns,nomega,omega,glatt0(:,:,:,1,:),flatdyn_moment(:,:,:,1,:),ntau,tau,flatdyn_t(:,:,:,1,:))
  call FLocDyn_F2T_v0(norb,ns,nomega,omega,glatt0(:,:,:,1,:),flatdyn_moment(:,:,:,1,:),ntau,tau,flatdyn_t2(:,:,:,1,:))  

  ! call FLocDyn_Normalization(norb,ns,ntau,tau,flatdyn_t(:,:,:,1,:))

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
