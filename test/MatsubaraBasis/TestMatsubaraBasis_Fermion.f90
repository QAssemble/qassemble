program TestMoment
  use Fourier
  use MatsubaraBasis
  use Common
  implicit none

  integer :: iomega, ik, is1,is2, ii,ntau,itau,nomega,itheta,norb,ns,nk, is,iorb,jorb,nc,l

  double precision :: beta, pi
  complex*16 :: err, errmax, ai

  double precision, allocatable :: omega(:), tau(:), eig(:)
  complex*16, allocatable :: &
    tempmat(:,:),tempmat1(:,:),tempmat2(:,:),eigmat(:,:),&
    hmat(:,:,:,:), identity(:,:),glatt0(:,:,:,:,:), flatdyn_t_ref(:,:,:,:,:), &
    flatdyn_t(:,:,:,:,:),flatdyn_t2(:,:,:,:,:),flatdyn_moment(:,:,:,:,:),flatdyn_high(:,:,:,:), &
    fout(:,:,:,:,:), fout2(:,:,:,:,:), &
    ftauout(:,:,:,:,:), chebyshev_coeff(:,:,:,:,:), chebyshev_coeff3(:,:,:,:,:), chebyshev_coeff2(:,:,:,:,:)
  


  nomega=3000
  ntau=nomega
  norb=2
  ns=2
  nk=2
  nc=100
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
      do iorb=1, norb
        do jorb=1, norb
          hmat(iorb,jorb,is,ik)=(iorb+jorb)*0.5d0+is*0.3d0*ik
          if (iorb .eq. jorb) then
            hmat(iorb,jorb,is,ik)=hmat(iorb,jorb,is,ik)-6.0d0
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
        do iorb=1, norb
          if (eig(iorb) .gt. 0.0d0) then
            eigmat(iorb,iorb)=-dexp(-eig(iorb)*tau(itau))*(1-1.0d0/(dexp(beta*eig(iorb))+1.0d0))
          else
            eigmat(iorb,iorb)=-dexp(eig(iorb)*(beta-tau(itau)))*(1.0d0/(dexp(beta*eig(iorb))+1.0d0))
          endif
        enddo
        flatdyn_t_ref(:,:,is,ik,itau)=matmul(matmul(tempmat, eigmat), transpose(dconjg(tempmat)))
      enddo
    enddo
  enddo


  allocate(chebyshev_coeff(norb,norb,ns,nk,0:(nc-1)))
  chebyshev_coeff=0.0d0
  allocate(chebyshev_coeff2(norb,norb,ns,nk,0:(nc-1)))
  chebyshev_coeff2=0.0d0
  allocate(chebyshev_coeff3(norb,norb,ns,nk,0:(nc-1)))
  chebyshev_coeff3=0.0d0
  
  allocate(ftauout(norb,norb,ns,nk,0:(ntau-1)))

  
  
  do ik=1, nk  
    do is=1, ns
      do iorb=1, norb
        do jorb=1, norb
          do l=0,  nc-1
            do itau=0, ntau-1    
              if (l .eq. 0) then
                chebyshev_coeff2(iorb,jorb,is,ik,l)=chebyshev_coeff2(iorb,jorb,is,ik,l)+1.0d0/dble(ntau)*flatdyn_t_ref(iorb,jorb,is,ik,itau)*dcos(l*dacos(2.0*tau(itau)/beta-1.0d0))
              else
                chebyshev_coeff2(iorb,jorb,is,ik,l)=chebyshev_coeff2(iorb,jorb,is,ik,l)+2.0d0/dble(ntau)*flatdyn_t_ref(iorb,jorb,is,ik,itau)*dcos(l*dacos(2.0*tau(itau)/beta-1.0d0))
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
  
  call FLatDyn_Cheby_Coeff(norb,ns,nk,ntau,tau,flatdyn_t_ref, nc,chebyshev_coeff)
  call FLatDyn_Cheby_Normalization(norb,ns,nk,nc,chebyshev_coeff, chebyshev_coeff3)

  call FLatDyn_Cheby_Tau(norb,ns,nk,nc,chebyshev_coeff, ntau,tau,ftauout)  

  open(unit=8, file='chebyshev.dat')
  errmax=0.0d0
  do ik=1, nk
    do is=1, ns
      do iorb=1, norb
        do jorb=1, norb
          do itau=0, nc-1
            err=chebyshev_coeff(iorb,jorb,is,ik,itau)-chebyshev_coeff3(iorb,jorb,is,ik,itau)
            if (cdabs(err) .gt. cdabs(errmax)) then
              errmax=err
            end if
            write(8, '(i5, 6f20.12)') itau, chebyshev_coeff(iorb,jorb,is,ik,itau), chebyshev_coeff3(iorb,jorb,is,ik,itau), err          
          enddo
          write(8,*) 
        end do
      end do
    enddo
  enddo
  close(8)
  print *, 'Flocdyn_Cheby_Coeff', errmax
  
  ! errmax=0.0d0
  ! open(unit=8, file='ftau.dat')
  ! do ik=1, nk
  !   do is=1, ns
  !     do iorb=1, norb
  !       do jorb=1, norb
  !         do itau=0, ntau-1          
  !           err=flatdyn_t_ref(iorb,jorb,is,ik,itau)-ftauout(iorb,jorb,is,ik,itau)
  !           write(8, '(i5, 6f20.12)') itau, flatdyn_t_ref(iorb,jorb,is,ik,itau), ftauout(iorb,jorb,is,ik,itau), err
  !           if (cdabs(err) .gt. cdabs(errmax)) then
  !             errmax=err
  !           end if
  !         enddo
  !       end do
  !     end do
  !   enddo
  ! enddo
  ! close(8)
  

  ! print *, 'Flocdyn_Cheby_tau', errmax

end program TestMoment
