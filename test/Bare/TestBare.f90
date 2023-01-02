program TestBare
    use Bare
    use Common
    use Fourier
    implicit none

    integer :: iorb, jorb, is, js, ik, ifreq, itau,itheta
    double precision :: omega(0:9), nu(0:9), tau1(0:9), tau2(0:9), w(4), beta , pi
    complex*16 :: fhlatt(4,4,2,5), bhlatt(4,4,2,2,5), fflatt(4,4,2,5,0:9), fflatt2(4,4,2,5,0:9), bflatt(4,4,2,2,5,0:9), bflatt2(4,4,2,2,5,0:9), &
                ftlatt(4,4,2,5,0:9), ftlatt2(4,4,2,5,0:9), btlatt(4,4,2,2,5,0:9), btlatt2(4,4,2,2,5,0:9),tempmat1(4,4), tempmat2(4,4), gfreq(0:9,4), &
                gtau(0:9,4), wfreq(0:9,4), wtau(0:9,4), fmoment(4,4,2,5,3), fhigh(4,4,2,5), bmoment(4,4,2,2,5,3), bhigh(4,4,2,2,5), err, ai

    ai = dcmplx(0.0d0, 1.0d0)
    beta=1.0d0/(8.617333262145d-5*300.0d0) 
    pi=datan2(1.0d0,1.0d0)*4.0d0 
!!!!!!!!!!!!! Fermion !!!!!!!!!!!!!!!
    fhlatt = 0.0d0
!!!!!!!!!!!!!!!!! Fermion Frequency !!!!!!!!!!!!!!!!!!!!

    omega = 0.0d0
    fflatt = 0.0d0
    fflatt2 = 0.0d0
    tempmat1 = 0.0d0
    tempmat2 = 0.0d0
    w = 0.0d0
    err = 0.0d0

    do ifreq = 0, 9                                     ! Fermion frequency condition
        omega(ifreq) = dcmplx(0.0d0, pi/beta*(2*ifreq+1))
    enddo

    do ik = 1, 5
        do is = 1, 2
            do iorb = 1, 4
                do jorb = 1, 4
                    if (iorb .eq. jorb) then
                        fhlatt(iorb, jorb, is, ik) = 1.0d0 + ik +is*0.1d0 + (iorb-jorb)*2.0d0
                    else
                        fhlatt(iorb, jorb, is, ik) = 0.1d0 + ik+ is*0.1d0 + (iorb-jorb)*0.1d0
                    endif
                enddo
            enddo
        enddo
    enddo

    do ik = 1,5
        do is = 1,2
            tempmat1 = fhlatt(:,:,is,ik)
            call hermitianeigen_dcmplx(4,w,tempmat1)

            gfreq = 0.0d0

            do iorb = 1, 4
                do ifreq = 0, 9
                    gfreq(ifreq, iorb) = 1.0d0/(ai*omega(ifreq)-w(iorb))
                enddo
            enddo

            do ifreq = 0, 9
                do iorb = 1, 4
                    do jorb = 1, 4
                        tempmat2(iorb, jorb) = tempmat1(iorb, jorb)*gfreq(ifreq, jorb)
                    enddo
                enddo
                call zgemm('n', 'c', 4, 4, 4, (1.0d0, 0.0d0), tempmat2, 4, tempmat1, 4, (0.0d0, 0.0d0), fflatt(1,1,is,ik,ifreq), 4)
            enddo
        enddo
    enddo

    call FLatFreq(4, 2, 5, 10, fhlatt,omega,fflatt2)

    print *, "FLatFreq"

    do ik = 1, 5
        do ifreq = 0, 9
            do is = 1, 2
                do iorb = 1, 4
                    do jorb = 1, 4
                        err = fflatt(iorb, jorb, is, ik, ifreq) - fflatt2(iorb, jorb, is, ik, ifreq)
                        if (cdabs(err) .gt. 1.0d-8) then
                            print '(5i5, 5f12.6)', iorb, jorb, is, ifreq, cdabs(err), fflatt(iorb, jorb, is, ik, ifreq), fflatt2(iorb, jorb, is, ik, ifreq)
                        endif
                    enddo
                enddo
            enddo
        enddo
    enddo

!!!!!!!!!!!!!!!!! Fermion Tau !!!!!!!!!!!!!!!!!!!!!!!!

    tau1 = 0.0d0
    ftlatt = 0.0d0
    ftlatt2 = 0.0d0
    tempmat1 = 0.0d0
    tempmat2 = 0.0d0
    w = 0.0d0
    err = 0.0d0
    itheta = 0.0d0
    fmoment = 0.0d0
    fhigh = 0.0d0

    do itau = 0, 9
        itheta = ttind(itau, 10)
        tau1(itau) = beta/2.0d0*(dcos(pi*(itheta+0.5d0)/dble(10))+1)
    enddo

    call FLatDyn_M(4,2,5,10,omega,fflatt,1,1,fmoment,fhigh)

    call FLatDyn_F2T(4,2,5,10,omega,fflatt,fmoment,10,tau1,ftlatt)

    call FLatTau(4,2,5,10,fhlatt,tau1,ftlatt2)

    print *, "FLatTau"

    do ik = 1, 5
        do itau = 0, 9
            do is = 1, 2
                do iorb = 1,4
                    do jorb = 1,4
                        err = ftlatt(iorb, jorb, is, ik, itau) - ftlatt2(iorb, jorb, is, ik, itau)
                        if (cdabs(err) .gt. 1.0d-8) then
                            print '(5i5, 5f12.6)', iorb, jorb, is, ik, itau, cdabs(err), ftlatt(iorb, jorb, is, ik, itau), ftlatt2(iorb, jorb, is, ik, itau)
                        endif
                    enddo
                enddo
            enddo
        enddo
    enddo

!!!!!!!!!!!!!!!!!! Boson !!!!!!!!!!!!!!!!!!!!!!!!
    bhlatt = 0.0d0
!!!!!!!!!!!!!!!!! Boson Frequency !!!!!!!!!!!!!!!!!!!!!!!!
    nu = 0.0d0
    bflatt = 0.0d0
    bflatt2 = 0.0d0
    tempmat1 = 0.0d0
    tempmat2 = 0.0d0
    w = 0.0d0
    err = 0.0d0

    do ifreq = 0,9
        nu(ifreq) = dcmplx(0.0d0, pi/beta*(2*ifreq))
    enddo

    do ik = 1, 5
        do is = 1, 2
            do js = 1, 2
                do iorb = 1, 4
                    do jorb = 1, 4
                        if (iorb .eq. jorb) then
                            bhlatt(iorb, jorb, is, js, ik) = 1.0d0 + ik + is*0.1d0 + (iorb - jorb)*2.0d0
                        else
                            bhlatt(iorb, jorb, is, js, ik) = 0.1d0 + ik + is*0.1d0 + (iorb - jorb)*0.1d0
                        endif
                    enddo
                enddo
            enddo
        enddo
    enddo

    do ik = 1, 5
        do is = 1, 2
            do js = 1, 2
                tempmat1 = bhlatt(:, :, is, js, ik)
                call hermitianeigen_dcmplx(4, w, tempmat1)

                wfreq = 0.0d0

                do iorb = 1, 4
                    do ifreq = 0, 9
                        wfreq(ifreq, iorb) = 1.0d0/(ai*nu(ifreq)-w(iorb))
                    enddo
                enddo

                do ifreq = 0, 9
                    do iorb = 1, 4
                        do jorb = 1, 4
                            tempmat2(iorb, jorb) = tempmat1(iorb, jorb)*wfreq(ifreq, jorb)
                        enddo
                    enddo
                    call zgemm('n', 'c', 4, 4, 4, (1.0d0, 0.0d0), tempmat2, 4, tempmat1, 4, (0.0d0, 0.0d0), bflatt(1,1,is,js,ik,ifreq),4)
                enddo
            enddo
        enddo
    enddo

    call BLatFreq(4,2,5,10,bhlatt,nu,bflatt2)

    print *, "BLatFreq"

    do ik = 1, 5
        do ifreq = 0, 9
            do is = 1, 2
                do js = 1, 2
                    do iorb = 1, 4
                        do jorb = 1, 4
                            err = bflatt(iorb, jorb, is, js, ik, ifreq) - bflatt2(iorb, jorb, is, js, ik, ifreq)
                            if (cdabs(err) .gt. 1.0d-8) then
                                print '(5i5, 5f12.6)', iorb, jorb, is, js, ik, ifreq, cdabs(err), bflatt(iorb, jorb, is, js, ik, ifreq), bflatt2(iorb, jorb, is, js, ik, ifreq)
                            endif
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo

!!!!!!!!!!!!!!!!! Boson Tau !!!!!!!!!!!!!!!!!!!!!!!!
    tau2 = 0.0d0
    btlatt = 0.0d0
    btlatt2 = 0.0d0
    tempmat1 = 0.0d0
    tempmat2 = 0.0d0
    w = 0.0d0
    err = 0.0d0
    itheta = 0.0d0
    bmoment = 0.0d0
    bhigh = 0.0d0

    do itau = 0, 9
        itheta = ttind(itau, 10)
        tau2(itheta) = beta/2.0d0*(dcos(pi*(itheta+0.5d0)/dble(10))+1)
    enddo

    call BLatDyn_M(4,2,5,10,nu,bflatt,0,1,bmoment,bhigh)

    call BLatDyn_F2T(4,2,5,10,nu,bflatt,bmoment,10,tau2,btlatt)

    call BLatTau(4,2,5,10,bhlatt,tau2,btlatt2)

    print *, "BLatTau"

    do ik = 1, 5
        do itau = 0, 9
            do is = 1, 2
                do js = 1, 2
                    do iorb = 1, 4
                        do jorb = 1, 4
                            err = btlatt(iorb, jorb, is, js, ik, itau) - btlatt2(iorb, jorb, is, js, ik, itau)
                            if (cdabs(err) .gt. 1.0d-6) then 
                                print '(5i5, 5f12.6)', iorb, jorb, is, js, ik, itau, cdabs(err), btlatt(iorb, jorb, is, js, ik, itau), btlatt2(iorb, jorb, is, js, ik, itau)
                            endif
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
    
end program TestBare

