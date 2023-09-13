program TestFinufft
    use Fourier
    use Common
    
    implicit none

    integer :: iorb, jorb, is, iomega, itau, itheta, norb, ns, nomega, ntau

    double precision :: beta, pi
    complex *16 :: ai, err, sum

    double precision, allocatable :: omega(:), tau(:)
    complex*16, allocatable :: fomega(:,:,:,:), ftau(:,:,:,:),moment(:,:,:,:),gfreq(:)

    norb = 1
    ns = 1
    nomega = 100
    ntau = nomega

    ai = dcmplx(0.0d0, 1.0d0)
    allocate(omega(0:(nomega-1)))
    omega = 0.0d0
    allocate(tau(0:(ntau-1)))
    tau = 0.0d0

    beta = 1.0d0/(8.617333262145d-5*300.0d0)
    pi = datan2(1.0d0,1.0d0)*4.0d0
    sum = 0.0d0

    allocate(fomega(norb,norb,ns,0:(nomega-1)))
    allocate(ftau(norb,norb,ns,0:(ntau-1)))
    allocate(moment(norb,norb,ns,3))
    allocate(gfreq(0:(nomega-1)))
    gfreq = 0.0d0

    do iomega = 0, nomega-1
        omega(iomega) = pi/beta*(2*iomega+1)
    enddo

    do iomega = 0, nomega-1
        gfreq(iomega) = 1.0d0/(ai*omega(iomega))
    enddo

    do iomega = 0, nomega-1
        sum = sum+gfreq(iomega)
    enddo
   
    sum = sum/nomega
    
    print *, "Frequency summation"
    print *, sum
    
    
    do iorb = 1, norb
        do jorb = 1, norb
           do is = 1, ns
                moment(iorb, jorb, is, 1) = 1.0d0
                moment(iorb, jorb, is, 2) = 0.0d0
                moment(iorb, jorb, is, 3) = 0.0d0
           enddo
        enddo
    enddo
    print *, moment 
    do iorb = 1, norb
        do jorb = 1, norb
            do is = 1, ns
                do iomega = 0, nomega-1
                     fomega(iorb, jorb, is, iomega) = gfreq(iomega)
                enddo
            enddo
        enddo
    enddo


    do itau = 1, ntau-1
        itheta=ttind(itau,ntau)
        tau(itau)=beta/2.0d0*(dcos(pi*(itheta+0.5d0)/dble(ntau))+1)
    enddo

    call FLocDyn_F2T(norb, ns, nomega, omega, fomega, moment, ntau, tau, ftau)

    print *, "Tau = beta"
    print *, ftau(1,1,1,ntau-1)
end program TestFinufft

    
