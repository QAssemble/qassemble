program TestBareLat
    use Bare
    use Common
    implicit none

    integer :: iorb, jorb, is, js, ik, ifreq
    double precision :: fomega(0:9), w(4), bomega(0:9)
    complex*16 :: fmat(4,4,2,5), gmat(4,4,2,5,0:9), gmat2(4,4,2,5,0:9), tempmat1(4,4), tempmat2(4,4), err, ai, hmat(4,4,2,5), &
                    gfreq(0:9,4), wmat(4,4,2,2,5,0:9), wmat2(4,4,2,2,5,0:9),hmat2(4,4,2,2,5), wfreq(0:9,4),tempmat3(4,4), tempmat4(4,4)
    
!!!!!!!!!!! Fermion Frequency !!!!!!!!!!!

    ai = dcmplx(0.0d0, 1.0d0)
    gmat = 0.0d0
    do ifreq = 0,9
        fomega(ifreq) = 2*ifreq+1
    enddo



    do ik = 1, 5
        do is = 1, 2
           tempmat1 = hmat(:,:,is,ik)
           call hermitianeigen_dcmplx(4, w, tempmat1)

           gfreq = 0.0d0


            do iorb = 1,4
                do ifreq = 0,9
                   gfreq(ifreq,iorb) = 1.0d0/(ai*fomega(ifreq)-w(iorb))
                enddo
            enddo

            do ifreq = 0, 9
                do iorb = 1, 4
                    do jorb = 1, 4
                       tempmat2(iorb, jorb) = tempmat1(iorb, jorb)*gfreq(ifreq, jorb)
                    enddo
                enddo
                call zgemm('n', 'c', 4, 4, 4, (1.0d0, 0.0d0), tempmat2, 4, tempmat1, 4, (0.0d0, 0.0d0), gmat(1,1,is,ik,ifreq), 4)
            enddo
        enddo
    enddo 

    call FLatFreq(4,2,5,10,hmat,fomega,gmat2)

    print *, 'Fermion Lattice Frequency'
    do ik = 1, 5
        do ifreq = 0,9
            do is = 1, 2
                do iorb = 1, 4
                    do jorb = 1, 4
                        err = gmat(iorb, jorb, is, ik, ifreq) - gmat2(iorb, jorb, is, ik, ifreq)
                        if (cdabs(err) .gt. 1.0d-8) then
                            print '(5i5, 5f12.6)', iorb, jorb, is, ifreq, cdabs(err), gmat(iorb, jorb, is, ik, ifreq), gmat2(iorb, jorb, is, ik, ifreq)
                        endif
                    enddo
                enddo
            enddo
        enddo
    enddo

!!!!!!!!!!! Bioson Frequency !!!!!!!!!!!

    wmat = 0.0d0
    do ifreq = 0,9
        bomega(ifreq) = 2*ifreq
    enddo

    do ik = 1, 5
        do is = 1, 2
            do js = 1, 2
                tempmat3 = hmat2(:,:,is,js,ik)
                call hermitianeigen_dcmplx(4,w,tempmat3)

                wfreq = 0.0d0

                do iorb = 1,4
                    do ifreq = 0,9
                        wfreq(ifreq, iorb) = 1.0d0/(ai*bomega(ifreq)-w(iorb))
                    enddo
                enddo

                do ifreq = 0, 9
                    do iorb = 1, 4
                        do jorb = 1,4
                            tempmat4(iorb, jorb) = tempmat3(iorb, jorb)*wfreq(ifreq, jorb)
                        enddo
                    enddo
                    call zgemm('n', 'c', 4,4,4, (1.0d0, 0.0d0), tempmat4, 4, tempmat3, 4, (0.0d0, 0.0d0), wmat(1, 1, is, js, ik, ifreq), 4)
                enddo
            enddo
        enddo
    enddo

    call BLatFreq(4,2,5,10,hmat2,bomega,wmat2)
    
    print *, 'Boson Lattice Frequency'
    do ik = 1, 5
        do ifreq = 0,9
            do is = 1,2
                do js = 1,2
                    do iorb = 1,4
                        do jorb = 1,4
                            err = wmat(iorb,jorb,is,js,ik,ifreq)-wmat2(iorb,jorb,is,js,ik,ifreq)
                            if (cdabs(err) .gt. 1.0d-8) then
                                print '(5i5, 5f12.6)', iorb,jorb,is,js,ik,ifreq,cdabs(err),wmat(iorb,jorb,is,js,ik,ifreq),wmat2(iorb,jorb,is,js,ik,ifreq)
                            endif
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo
    


end program TestBareLat



