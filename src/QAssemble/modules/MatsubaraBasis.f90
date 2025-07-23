Module MatsubaraBasis

  implicit None
  private :: gcoeff
  Public :: &
    F_Chebyshev_Coeff, &    
    FLocDyn_Chebyshev_Coeff, &
    FLatDyn_Chebyshev_Coeff, &
    
    F_Chebyshev_Tau, &    
    FLocDyn_Chebyshev_Tau, &
    FLatDyn_Chebyshev_Tau,&

    F_Chebyshev_1stMoment, &
    FLocDyn_Chebyshev_Normalization, &
    FLatDyn_Chebyshev_Normalization, &
    
    LegendreC2ChebyshevC, &
    ChebyshevC2LegendreC, &
    LegendreConvolutionMatrix, &
    ChebyshevConvolutionMatrix, &
    Legendre2Chebyshev


contains

  subroutine Legendre2Chebyshev(nc,transmat)
    implicit none
    integer, intent(in) :: nc
    complex*16, intent(out) :: transmat(0:(nc-1), 0:(nc-1))

    integer :: ic, jc,kc

    ! double precision, external :: gcoeff

    transmat=0.0d0

    do ic=0, nc-1
      do jc=0, ic
        kc=iabs(2*jc-ic)
        transmat(ic,kc)=transmat(ic,kc)+gcoeff(jc)*gcoeff(ic-jc)
      enddo
    enddo
! transpose of the original equation
    transmat=transpose(transmat)
    
  end subroutine Legendre2Chebyshev


  DOUBLE PRECISION function gcoeff(m)
    IMPLICIT NONE
    integer, intent(in) :: m

    integer :: ii
    
    if (m<0) print *, &
      "gcoeff defined only for non-negative numbers!"
    
    if (m .eq. 0) then
      gcoeff=1.0d0
    else
      gcoeff=1.0d0      
      do ii=1, m
        gcoeff=gcoeff*(2*ii-1)
      enddo
      
      do ii=1, m
        gcoeff=gcoeff/(ii*2)
      enddo
    endif
  end function gcoeff
  

  

  subroutine ChebyshevConvolutionMatrix(nc, chebyshev_coeff, chebyshev_matrix, fbsign)
    implicit none

    integer, intent(in) :: nc, fbsign
    complex*16, intent(in) :: chebyshev_coeff(0:(nc-1))
    complex*16, intent(out) :: chebyshev_matrix(0:(nc-1),0:(nc-1))

    complex*16 :: legendre_coeff(0:(nc-1)), legendre_matrix(0:(nc-1),0:(nc-1)), tempmat(0:(nc-1),0:(nc-1)), transmat(0:(nc-1),0:(nc-1)), transmatinv(0:(nc-1),0:(nc-1))


    legendre_coeff=0.0d0
    legendre_matrix=0.0d0
    tempmat=0.0d0

    call ChebyshevC2LegendreC(nc, chebyshev_coeff,legendre_coeff)

    call LegendreConvolutionMatrix(nc, legendre_coeff, legendre_matrix, fbsign)

    call Legendre2Chebyshev(nc,transmat)

    call dcmplx_matinv(transmat, transmatinv, nc, nc)

    call zgemm('n','n',nc,nc,nc,(1.0d0,0.0d0),transmat,nc,legendre_matrix,nc,(0.0d0,0.0d0),tempmat,nc)
    call zgemm('n','n',nc,nc,nc,(1.0d0,0.0d0),tempmat,nc,transmatinv,nc,(0.0d0,0.0d0),chebyshev_matrix,nc)
  end subroutine ChebyshevConvolutionMatrix

  subroutine LegendreConvolutionMatrix(nc, legendre_coeff, legendre_matrix, fbsign)
    implicit none

    integer, intent(in) :: nc, fbsign
    complex*16, intent(in) :: legendre_coeff(0:(nc-1))
    complex*16, intent(out) :: legendre_matrix(0:(nc-1),0:(nc-1))

    integer :: ic, jc,ii, interval_sign
    complex*16 :: lmatrix(0:(nc-1), 0:(nc-1), 2)

    legendre_matrix=0.0d0

    lmatrix=0.0d0

    do ii=1, 2 ! 1: <, 2:>
      if (ii .eq. 1) then
        interval_sign=1
      else
        interval_sign=-1
      endif

! jc=0
      do ic=0, nc-1
        if (ic .eq. 0) then
          lmatrix(ic,0,ii)=legendre_coeff(0)+interval_sign*legendre_coeff(1)/3.0d0
        else
          lmatrix(ic,0,ii)=interval_sign*(legendre_coeff(ic-1)/dble(2*ic-1)-legendre_coeff(ic+1)/dble(2*ic+3))
        endif
      enddo

! jc=1
      do ic=0, nc-1
        if (ic .eq. 0) then
          lmatrix(ic,1,ii)=-interval_sign*lmatrix(1,0,ii)/3.0d0
        else
          lmatrix(ic,1,ii)=-interval_sign*lmatrix(ic,0,ii)+lmatrix(ic-1,0,ii)/dble(2*ic-1)-lmatrix(ic+1,0,ii)/dble(2*ic+3)
        endif
      enddo

      do jc=2, nc-1      
        do ic=jc, nc-1
          if (ic .ne. nc-1) then
            lmatrix(ic,jc,ii) &
              =-lmatrix(ic+1,jc-1,ii)*dble(2*jc-1)/dble(2*ic+3) &
              +lmatrix(ic-1,jc-1,ii)*dble(2*jc-1)/dble(2*ic-1) &
              +lmatrix(ic,jc-2,ii)
          else
            lmatrix(ic,jc,ii) &
              =lmatrix(ic-1,jc-1,ii)*dble(2*jc-1)/dble(2*ic-1) &
              +lmatrix(ic,jc-2,ii) 
          endif
        enddo
      enddo

      do ic=0, nc-1
        do jc=ic+1, nc-1
          lmatrix(ic,jc,ii)=(-1)**(ic+jc)*lmatrix(jc,ic,ii)*dble(2*ic+1)/dble(2*jc+1)
        enddo
      enddo
    enddo

    legendre_matrix=lmatrix(:,:,1)+lmatrix(:,:,2)*fbsign

  end subroutine LegendreConvolutionMatrix

  subroutine LegendreC2ChebyshevC(nc, legendre_coeff, chebyshev_coeff)

    implicit none
    integer, intent(in) :: nc
    complex*16, intent(in) :: legendre_coeff(0:(nc-1))
    complex*16, intent(out) :: chebyshev_coeff(0:(nc-1))

    complex*16 :: transmat(0:(nc-1), 0:(nc-1))


    transmat=0.0d0
    chebyshev_coeff=0.0d0
    call Legendre2Chebyshev(nc,transmat)
    call zgemv('N',nc,nc,(1.0d0, 0.0d0), transmat, nc, legendre_coeff, 1,(0.0d0, 0.0d0), chebyshev_coeff, 1)

  end subroutine LegendreC2ChebyshevC


  subroutine ChebyshevC2LegendreC(nc, chebyshev_coeff,legendre_coeff)

    implicit none
    integer, intent(in) :: nc
    complex*16, intent(in) :: chebyshev_coeff(0:(nc-1))
    complex*16, intent(out) :: legendre_coeff(0:(nc-1))    

    complex*16 :: transmat(0:(nc-1), 0:(nc-1)), transmatinv(0:(nc-1), 0:(nc-1))


    transmat=0.0d0
    transmatinv=0.0d0    
    legendre_coeff=0.0d0
    call Legendre2Chebyshev(nc,transmat)
    call dcmplx_matinv(transmat, transmatinv, nc, nc)
    call zgemv('N',nc,nc,(1.0d0, 0.0d0), transmatinv, nc, chebyshev_coeff, 1,(0.0d0, 0.0d0), legendre_coeff, 1)

  end subroutine ChebyshevC2LegendreC


  subroutine F_Chebyshev_1stMoment(nc, chebyshev_coeff_in, normalization)

    implicit none

    integer, intent(in) :: nc
    complex*16, intent(in) :: chebyshev_coeff_in(0:(nc-1))
    complex*16, intent(out) :: normalization


    integer :: ic
    
    normalization =0.0d0
    do ic=0, nc-1, 2
      normalization=normalization-chebyshev_coeff_in(ic)*2.0d0
    enddo

  end subroutine F_Chebyshev_1stMoment


  subroutine FLocDyn_Chebyshev_Normalization(norb,ns, nc, chebyshev_coeff_in, chebyshev_coeff_out)

    implicit none

    integer, intent(in) :: norb,ns,nc
    complex*16, intent(in) :: chebyshev_coeff_in(norb,norb,ns,0:(nc-1))
    complex*16, intent(out) :: chebyshev_coeff_out(norb,norb,ns,0:(nc-1))        

    integer :: is,iorb,jorb,ic
    complex*16 :: normalization, normalization_sum

    chebyshev_coeff_out=0.0d0
    do is=1, ns
      normalization_sum=0.0d0
      do iorb=1, norb
        jorb=iorb
        call F_Chebyshev_1stMoment(nc,chebyshev_coeff_in(iorb,jorb,is,:), normalization)
        normalization_sum=normalization_sum+normalization
      enddo
      chebyshev_coeff_out(:,:,is,:)=chebyshev_coeff_in(:,:,is,:)/(normalization_sum/norb)
    enddo

  end subroutine FLocDyn_Chebyshev_Normalization


  subroutine FLatDyn_Chebyshev_Normalization(norb,ns,nk,nc,chebyshev_coeff_in,chebyshev_coeff_out)
    implicit none

    integer, intent(in) :: norb,ns,nc,nk
    complex*16, intent(in) :: chebyshev_coeff_in(norb,norb,ns,nk,0:(nc-1))
    complex*16, intent(out) :: chebyshev_coeff_out(norb,norb,ns,nk,0:(nc-1))    

    integer :: ik


    chebyshev_coeff_out=0.0d0

    do ik=1, nk
      call FLocDyn_Chebyshev_Normalization(norb,ns,nc,chebyshev_coeff_in(:,:,:,ik,:),chebyshev_coeff_out(:,:,:,ik,:))
    enddo
  end subroutine FLatDyn_Chebyshev_Normalization


  subroutine F_Chebyshev_Coeff(ntau,tau,ftau,nc,chebyshev_coeff)
! use common, only: ttind
    implicit none

    integer, intent(in) :: ntau,nc
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: ftau(0:(ntau-1))
    complex*16, intent(out) :: chebyshev_coeff(0:(nc-1))


    integer :: itheta, itau
    integer :: ierr    

    integer*8 :: ntau_finu, nc_finu    
    integer*8, allocatable :: null

    double precision :: theta((-ntau):(ntau-1)), pi, beta
    complex*16 :: ftheta((-ntau):(ntau-1)), chebyshev_temp((-nc+1):(nc-1))

! integer, external :: ttind

    if (nc .gt. ntau) then
      print *, 'too many coefficient has been requested !'
      stop
    endif

    pi=datan2(1.0d0,1.0d0)*4.0d0

    beta=tau(0)*2.0d0/(dcos(pi*(ntau-0.5d0)/dble(ntau))+1.0d0)    

    do itheta=0, ntau-1
      itau=ttind(itheta, ntau)
      theta(itheta)=dacos(2.0d0/beta*tau(itau)-1)
      theta(-itheta-1)=-theta(itheta)      
    enddo

    ntau_finu=2*ntau
    nc_finu=2*nc-1

    chebyshev_coeff=0.0d0

    ftheta=0.0d0
    chebyshev_temp=0.0d0
    do itheta=0, ntau-1
      itau=ttind(itheta, ntau)            
      ftheta(itheta)=ftau(itau)
      ftheta(-itheta-1)=ftheta(itheta)
    enddo
    call finufft1d1(ntau_finu,theta,ftheta,1, 1.0d-12, nc_finu,chebyshev_temp,null,ierr)    
    chebyshev_coeff=chebyshev_temp(0:(nc-1))/ntau
    chebyshev_coeff(0)=chebyshev_coeff(0)/2.0d0
  end subroutine F_Chebyshev_Coeff


  subroutine FLocDyn_Chebyshev_Coeff(norb,ns,ntau,tau,ftau,nc,chebyshev_coeff)
! use common, only: ttind
    implicit none

    integer, intent(in) :: ntau,norb,ns,nc
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: ftau(norb,norb,ns,0:(ntau-1))
    complex*16, intent(out) :: chebyshev_coeff(norb,norb,ns,0:(nc-1))


    integer :: is, iorb,jorb

! integer, external :: ttind

    chebyshev_coeff=0.0d0

    do is=1, ns
      do iorb=1, norb
        do jorb=1, norb
          call F_Chebyshev_Coeff(ntau,tau,ftau(iorb,jorb,is,:),nc,chebyshev_coeff(iorb,jorb,is,:))
        enddo
      enddo
    enddo
  end subroutine FLocDyn_Chebyshev_Coeff



  subroutine FLatDyn_Chebyshev_Coeff(norb,ns,nk,ntau,tau,ftau,nc,chebyshev_coeff)
! use common, only: ttind
    implicit none

    integer, intent(in) :: ntau,norb,ns,nc,nk
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: ftau(norb,norb,ns,nk,0:(ntau-1))
    complex*16, intent(out) :: chebyshev_coeff(norb,norb,ns,nk,0:(nc-1))


    integer :: ik

    if (nc .gt. ntau) then
      print *, 'too many coefficient has been requested !'
      stop
    endif

    chebyshev_coeff=0.0d0

    do ik=1, nk
      call FLocDyn_Chebyshev_Coeff(norb,ns,ntau,tau,ftau(:,:,:,ik,:),nc,chebyshev_coeff(:,:,:,ik,:))
    enddo

  end subroutine FLatDyn_Chebyshev_Coeff


  subroutine F_Chebyshev_Tau(nc, chebyshev_coeff, ntau,tau,ftau)

    implicit none

    integer, intent(in) :: ntau,nc
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: chebyshev_coeff(0:(nc-1))
    complex*16, intent(out) :: ftau(0:(ntau-1))    

    integer :: itheta, itau, ic
    integer :: ierr    

    integer*8 :: ntau_finu, nc_finu    
    integer*8, allocatable :: null

    complex*16 :: chebyshev_array((-nc+1):(nc-1)),ftau_temp(0:(ntau-1))
    double precision :: theta(0:(ntau-1)), pi, beta

    ftau=0.0d0

    pi=datan2(1.0d0,1.0d0)*4.0d0

    beta=tau(0)*2.0d0/(dcos(pi*(ntau-0.5d0)/dble(ntau))+1.0d0)

    ntau_finu=ntau
    nc_finu=2*nc-1

    do itau=0, ntau-1
      itheta=ttind(itau, ntau)
      theta(itheta)=dacos(2.0d0/beta*tau(itau)-1)
    enddo

    chebyshev_array=0.0d0
    do ic=-nc+1, nc-1
      if (ic .eq. 0) then
        chebyshev_array(ic)=chebyshev_coeff(ic)
      elseif (ic .gt. 0) then
        chebyshev_array(ic)=chebyshev_coeff(ic)/2.0d0
      else
        chebyshev_array(ic)=chebyshev_coeff(-ic)/2.0d0
      endif
    enddo
    call finufft1d2(ntau_finu,theta,ftau_temp,-1, 1.0d-12, nc_finu,chebyshev_array,null,ierr)
    do itau=0, ntau-1
      itheta=ttind(itau, ntau)            
      ftau(itau) =ftau_temp(itheta)
    enddo
  end subroutine F_Chebyshev_Tau


  subroutine FLocDyn_Chebyshev_Tau(norb,ns,nc, chebyshev_coeff, ntau,tau,ftau)

    implicit none

    integer, intent(in) :: ntau,norb,ns,nc
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: chebyshev_coeff(norb,norb,ns,0:(nc-1))
    complex*16, intent(out) :: ftau(norb,norb,ns,0:(ntau-1))    

    integer :: is, iorb,jorb

    ftau=0.0d0

    do is=1, ns
      do iorb=1, norb
        do jorb=1, norb

          call F_Chebyshev_Tau(nc, chebyshev_coeff(iorb,jorb,is,:), ntau,tau,ftau(iorb,jorb,is,:))
        enddo
      enddo
    enddo


  end subroutine FLocDyn_Chebyshev_Tau


  subroutine FLatDyn_Chebyshev_Tau(norb,ns,nk,nc, chebyshev_coeff, ntau,tau,ftau)

    implicit none

    integer, intent(in) :: ntau,norb,ns,nc,nk
    double precision, intent(in) :: tau(0:(ntau-1))
    complex*16, intent(in) :: chebyshev_coeff(norb,norb,ns,nk,0:(nc-1))
    complex*16, intent(out) :: ftau(norb,norb,ns,nk,0:(ntau-1))

    integer :: ik

    ftau=0.0d0
    do ik=1, nk
      call FLocDyn_Chebyshev_Tau(norb,ns,nc, chebyshev_coeff(:,:,:,ik,:), ntau,tau,ftau(:,:,:,ik,:))
    enddo
  end subroutine FLatDyn_Chebyshev_Tau

end Module MatsubaraBasis


