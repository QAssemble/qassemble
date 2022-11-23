program TestMoment
  use Fourier
  use Common
  implicit none

  integer :: ik, is1,is2,ii,ntau,itau,itheta,nnu, norb,ns,nk, nn1(2), nn2(2),ind1,ind2,inu,ndim,is,js,iorb,jorb

  double precision :: beta, pi
  complex*16 :: err, errmax, ai

  double precision, allocatable :: nu(:), tau(:), eig(:)
  complex*16, allocatable :: &
    tempmat(:,:),tempmat1(:,:),tempmat2(:,:),eigmat(:,:),identity(:,:),&    
    wmat(:,:,:), wlatt0(:,:,:,:,:,:), blatdyn_t_ref(:,:,:,:,:,:), &
    blatdyn_t(:,:,:,:,:,:),blatdyn_t2(:,:,:,:,:,:),blatdyn_moment(:,:,:,:,:,:),blatdyn_high(:,:,:,:,:)  


  nnu=1000
  ntau=nnu
  norb=3
  ns=2
  nk=2
  ai=dcmplx(0.0d0, 1.0d0)


  allocate(nu(0:(nnu-1)))
  nu=0.0d0

  allocate(tau(0:(ntau-1)))
  tau=0.0d0


  beta=1.0d0/(8.617333262145d-5*300.0d0)
  pi=datan2(1.0d0,1.0d0)*4.0d0

  do inu=0, nnu-1
    nu(inu)=pi/beta*(2*inu)
  enddo

  do itau=0, ntau-1
    itheta=ttind(itau,ntau)
    tau(itau)=beta/2.0d0*(dcos(pi*(itheta+0.5d0)/dble(ntau))+1)    
  enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ndim=norb*ns

  allocate(eig(ndim))
  eig=0.0d0
  allocate(eigmat(ndim,ndim))
  eigmat=0.0d0

  allocate(identity(ndim,ndim))
  identity=0.0d0
  do ii=1, ndim
    identity(ii,ii)=1.0d0
  enddo
  allocate(tempmat(ndim,ndim))
  tempmat=0.0d0
  allocate(tempmat1(ndim,ndim))
  tempmat=0.0d0
  allocate(tempmat2(ndim,ndim))
  tempmat2=0.0d0

  allocate(wmat(ndim,ndim,nk))
  wmat=0.0d0
  allocate(wlatt0(norb,norb,ns,ns,nk,0:(nnu-1)))
  wlatt0=0.0d0
  allocate(blatdyn_t_ref(norb,norb,ns,ns,nk,0:(ntau-1)))
  blatdyn_t_ref=0.0d0
  allocate(blatdyn_t(norb,norb,ns,ns,nk,0:(ntau-1)))
  blatdyn_t=0.0d0
  allocate(blatdyn_t2(norb,norb,ns,ns,nk,0:(ntau-1)))
  blatdyn_t2=0.0d0  
  allocate(blatdyn_moment(norb,norb,ns,ns,nk,3))
  blatdyn_moment=0.0d0  
  allocate(blatdyn_high(norb,norb,ns,ns,nk))
  blatdyn_high=0.0d0  

  do ik=1, nk
    do is=1, ns
      do js=1, ns      
        do iorb=1, norb
          do jorb=1, norb
            nn1=(/iorb,is/)
            nn2=(/jorb,js/)
            call indexing(ndim,2,(/norb,ns/),1,ind1,nn1)            
            call indexing(ndim,2,(/norb,ns/),1,ind2,nn2)            
            if ((is .eq. js) .and. (iorb .eq. jorb)) then
              wmat(ind1,ind2,ik)=(iorb+jorb-5)*0.5d0+(is+js-2)*0.5d0+ik*0.8d0-2.0d0-0.1d0
            else
              wmat(ind1,ind2,ik)=(iorb+jorb)*0.01d0+(is+js)*0.01d0+ik*0.01d0              
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  do ik=1, nk 
    do inu=0, nnu-1
      tempmat1=0.0d0
      tempmat1=identity*nu(inu)*ai-wmat(:,:,ik)
      call dcmplx_matinv(tempmat1, tempmat2,ndim,ndim)
      do is=1, ns        
        do js=1, ns      
          do iorb=1, norb
            do jorb=1, norb
              nn1=(/iorb,is/)
              nn2=(/jorb,js/)
              call indexing(ndim,2,(/norb,ns/),1,ind1,nn1)            
              call indexing(ndim,2,(/norb,ns/),1,ind2,nn2)                  
              wlatt0(iorb,jorb,is,js,ik,inu)=tempmat2(ind1,ind2)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

! do inu=0, nnu-1
!   print '(6f20.12)', wlatt0(1,1,1,1,1,inu)
! enddo

  do ik=1, nk
    tempmat=wmat(:,:,ik)
    call hermitianeigen_dcmplx(ndim,eig,tempmat)
    print '(i6, 100f20.12)', ik, eig(:)
    
    eigmat=0.0d0

    do itau=0, ntau-1                
      do iorb=1, ndim
        if (eig(iorb) .gt. 0.0d0) then
          eigmat(iorb,iorb)=-dexp(-eig(iorb)*tau(itau))*(1-1.0d0/(dexp(beta*eig(iorb))-1.0d0))
        else
          eigmat(iorb,iorb)=-dexp(eig(iorb)*(beta-tau(itau)))*(1.0d0/(dexp(beta*eig(iorb))-1.0d0))
        endif
      enddo
      tempmat2=matmul(matmul(tempmat, eigmat), transpose(dconjg(tempmat)))
      do is=1, ns        
        do js=1, ns      
          do iorb=1, norb
            do jorb=1, norb
              nn1=(/iorb,is/)
              nn2=(/jorb,js/)
              call indexing(ndim,2,(/norb,ns/),1,ind1,nn1)            
              call indexing(ndim,2,(/norb,ns/),1,ind2,nn2)            

              blatdyn_t_ref(iorb,jorb,is,js,ik,itau)=tempmat2(ind1,ind2)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo


  call BLocDyn_M(norb,ns,nnu,nu,wlatt0(:,:,:,:,1,:),0,1,blatdyn_moment(:,:,:,:,1,:),blatdyn_high(:,:,:,:,1))


! print *, blatdyn_moment(:,:,:,:,1,:)

  call BLocDyn_F2T(norb,ns,nnu,nu,wlatt0(:,:,:,:,1,:),blatdyn_moment(:,:,:,:,1,:),ntau,tau,blatdyn_t(:,:,:,:,1,:))
  call BLocDyn_F2T_v0(norb,ns,nnu,nu,wlatt0(:,:,:,:,1,:),blatdyn_moment(:,:,:,:,1,:),ntau,tau,blatdyn_t2(:,:,:,:,1,:))

  errmax=0.0d0
  do is=1, ns
    do js=1, ns      
      do iorb=1, norb
        do jorb=1, norb
          do itau=0, ntau-1
            ! err=blatdyn_t(iorb,jorb,is,js,1,itau)-blatdyn_t_ref(iorb,jorb,is,js,1,itau)
            err=blatdyn_t(iorb,jorb,is,js,1,itau)-blatdyn_t2(iorb,jorb,is,js,1,itau)            
            if (cdabs(err) .gt. cdabs(errmax)) then
              errmax=err
            endif
          end do
        enddo
      end do
    end do
  enddo
  print *, 'BLocDyn_F2T' , errmax  

! open(unit=5, file='btau.dat')
!   do is=1, ns
!     do js=1, ns      
!       do iorb=1, norb
!         do jorb=1, norb
!           do itau=0, ntau-1            
!             write(5, '(7(2x,f12.6))') tau(itau),blatdyn_t(iorb,jorb,is,js,1,itau), blatdyn_t_ref(iorb,jorb,is,js,1,itau), blatdyn_t(iorb,jorb,is,js,1,itau)-blatdyn_t_ref(iorb,jorb,is,js,1,itau)
!           end do
!         enddo
!       end do
!     end do
!   enddo
!   close(5)
  

  call BLatDyn_M(norb,ns,nk,nnu,nu,wlatt0,0,0,blatdyn_moment,blatdyn_high)

! print *, blatdyn_moment(:,:,:,:,1,:)

  call BLatDyn_F2T(norb,ns,nk,nnu,nu,wlatt0,blatdyn_moment,ntau,tau,blatdyn_t)
  call BLatDyn_F2T_v0(norb,ns,nk,nnu,nu,wlatt0,blatdyn_moment,ntau,tau,blatdyn_t2)  

  errmax=0.0d0
  do ik=1, nk
    do is=1, ns
      do js=1, ns      
        do iorb=1, norb
          do jorb=1, norb
            do itau=0, ntau-1
              ! err=blatdyn_t(iorb,jorb,is,js,ik,itau)-blatdyn_t_ref(iorb,jorb,is,js,ik,itau)
              err=blatdyn_t(iorb,jorb,is,js,ik,itau)-blatdyn_t2(iorb,jorb,is,js,ik,itau)              
              if (cdabs(err) .gt. cdabs(errmax)) then
                print '(6i5, f20.12, 2(2e20.12, 2x))', ik,is,js,iorb,jorb,itau, tau(itau), blatdyn_t(iorb,jorb,is,js,ik,itau), blatdyn_t_ref(iorb,jorb,is,js,ik,itau)                
                errmax=err
              endif
            end do
          enddo
        end do
      end do
    enddo
  enddo
  print *, 'BLatDyn_F2T' , errmax    

! ! errmax=0.0d0
!   open(unit=5, file='btau.dat')
!   do ik=1, nk
!     do is=1, ns
!       do js=1, ns      
!         do iorb=1, norb
!           do jorb=1, norb
!             do itau=0, ntau-1
!               write(5, '(7(2x,e12.6))') tau(itau),blatdyn_t(iorb,jorb,is,js,ik,itau), blatdyn_t_ref(iorb,jorb,is,js,ik,itau), blatdyn_t(iorb,jorb,is,js,ik,itau)-blatdyn_t_ref(iorb,jorb,is,js,ik,itau)
!             end do
!           enddo
!         end do
!       end do
!     enddo
!   enddo
!   close(5)





end program TestMoment
