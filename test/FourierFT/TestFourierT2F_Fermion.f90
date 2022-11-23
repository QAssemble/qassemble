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
    flatdyn_t(:,:,:,:,:),flatdyn_t2(:,:,:,:,:),flatdyn_moment(:,:,:,:,:),flatdyn_high(:,:,:,:), &
    fout(:,:,:,:,:), fout2(:,:,:,:,:)


  nomega=3000
  ntau=nomega
  norb=3
  ns=2
  nk=3
  ai=dcmplx(0.0d0, 1.0d0)

  allocate(tempmat(norb,norb))
  allocate(tempmat1(norb,norb))
  allocate(tempmat2(norb,norb))
  allocate(eig(norb))
  allocate(eigmat(norb,norb))
  allocate(hmat(norb,norb,ns,nk))
  allocate(identity(norb,norb))
  allocate(glatt0(norb,norb,ns,nk,0:(nomega-1)))
  allocate(fout(norb,norb,ns,nk,0:(nomega-1)))
  allocate(fout2(norb,norb,ns,nk,0:(nomega-1)))    
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

  call FLocDyn_T2F(norb,ns,ntau, tau, flatdyn_t_ref(:,:,:,1,:), nomega,omega,fout(:,:,:,1,:))
  call FLocDyn_T2F_v0(norb,ns,ntau, tau, flatdyn_t_ref(:,:,:,1,:), nomega,omega,fout2(:,:,:,1,:))  

  open(unit=8, file='fomega.dat')
  errmax=0.0d0
  do iomega=0, nomega-1
    do is=1, ns
      do iorb1=1, norb
        do iorb2=1, norb

          ! err=glatt0(iorb1,iorb2,is,1,iomega)-fout(iorb1,iorb2,is,1,iomega)
          err=fout(iorb1,iorb2,is,1,iomega)-fout2(iorb1,iorb2,is,1,iomega)                    
          ! write(8, '(i7, 10(f20.12))') iomega, glatt0(iorb1,iorb2,is,1,iomega), fout(iorb1,iorb2,is,1,iomega), err          
          if (cdabs(err) .gt. cdabs(errmax)) then
            errmax=err
          end if
        enddo
      end do
    end do
  enddo
  close(8)
  print *, 'Flocdyn_T2F', errmax



  call FLatDyn_T2F(norb,ns,nk,ntau, tau, flatdyn_t_ref, nomega,omega,fout)
  call FLatDyn_T2F_v0(norb,ns,nk,ntau, tau, flatdyn_t_ref, nomega,omega,fout2) 

  open(unit=8, file='fomega.dat')
  errmax=0.0d0
  do iomega=0, nomega-1
    do ik=1, nk
      do is=1, ns
        do iorb1=1, norb
          do iorb2=1, norb
            
! err=glatt0(iorb1,iorb2,is,1,iomega)-fout(iorb1,iorb2,is,1,iomega)
            err=fout(iorb1,iorb2,is,ik,iomega)-fout2(iorb1,iorb2,is,ik,iomega)                    
! write(8, '(i7, 10(f20.12))') iomega, glatt0(iorb1,iorb2,is,1,iomega), fout(iorb1,iorb2,is,1,iomega), err          
            if (cdabs(err) .gt. cdabs(errmax)) then
              errmax=err
            end if
          enddo
        end do
      end do
    enddo
  enddo
  close(8)
  print *, 'Flatdyn_T2F', errmax  

end program TestMoment
