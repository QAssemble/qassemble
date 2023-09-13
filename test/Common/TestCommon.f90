program TestCommon
    use Common
    implicit none

    integer :: norb, ns, nk, iorb, jorb, korb, is
    complex*16, allocatable :: mat1(:,:), invmat(:,:), mat2(:,:), mat1d_1(:), mat1d_2(:), mat2d_1(:,:), mat2d_2(:,:), &
                               mat3d_1(:,:,:), mat3d_2(:,:,:), tempmat1(:), tempmat2(:,:), tempmat3(:,:,:)
    double precision :: w(4)
    complex*16 :: err
    integer*8 :: ind, temp(2)

    norb = 4
    ns = 2
    w = 0.0d0


    allocate(mat1(norb,norb))
    allocate(mat2(norb, norb))
    allocate(invmat(norb,norb))

    allocate(mat1d_1(norb))
    allocate(mat1d_2(norb))
    allocate(tempmat1(norb))
    allocate(mat2d_1(norb,norb))
    allocate(mat2d_2(norb,norb))
    allocate(tempmat2(norb,norb))
    allocate(mat3d_1(norb,norb,norb))
    allocate(mat3d_2(norb,norb,norb))
    allocate(tempmat3(norb,norb,norb))

    mat1 = 0.0d0
    mat2 = 0.0d0
    invmat = 0.0d0
    mat1d_1 = 0.0d0
    mat1d_2 = 0.0d0
    mat2d_1 = 0.0d0
    mat2d_2 = 0.0d0
    mat3d_1 = 0.0d0
    mat3d_2 = 0.0d0
    tempmat1 = 0.0d0
    tempmat2 = 0.0d0
    tempmat3 = 0.0d0

    err = 0.0d0

    do iorb = 1, norb
        do jorb = 1, norb
            if (iorb .eq. jorb) then
                mat1(iorb, jorb) = 1.0d0 
            else
                mat1(iorb,jorb) = -1.0d0
            endif
        enddo
    enddo


    call dcmplx_matinv(mat1, invmat, norb, norb)

    call dcmplx_matinv(invmat, mat2, norb, norb)

    do iorb = 1, norb
        do jorb = 1, norb
            err = mat1(iorb, jorb) - mat2(iorb, jorb)
            if (cdabs(err) .gt. 1.0d-8) then
                print *, iorb, jorb, cdabs(err), mat1(iorb,jorb), mat2(iorb, jorb)
            endif
        enddo
    enddo


    call hermitianeigen_dcmplx(norb, w, mat1)
    
    print *, "Hermitianeigen"
    print *, w

    print *, "FFTW3_1D"

    mat1d_1 = 1.0d0
    
    mat1d_2 = mat1d_1
    call fftw3_1d(mat1d_1,norb,1)
    call fftw3_1d(mat1d_1,norb,1)
    
    mat1d_1 = mat1d_1/norb
    err = 0.0d0

    do iorb = 1, norb
        err = mat1d_2(iorb) - mat1d_1(iorb)
        if (cdabs(err) .gt. 1.0d-8) then
            print *, iorb, cdabs(err), mat1d_2(iorb), mat1d_1(iorb)
        endif
    enddo

    print *, "FFTW3_2D"

    mat2d_1 = 1.0d0

    mat2d_2 = mat2d_1
    call fftw3_2d(mat2d_1,norb,norb,1)
    call fftw3_2d(mat2d_1,norb,norb,-1)

    mat2d_1 = mat2d_1/(norb)**2

    err = 0.0d0

    do iorb = 1, norb
        do jorb = 1, norb
            err = mat2d_2(iorb, jorb) - mat2d_1(iorb, jorb)
            if (cdabs(err) .gt. 1.0d-8) then
                print *, iorb, jorb, cdabs(err), mat2d_1(iorb, jorb), mat2d_2(iorb, jorb)
            endif
        enddo
    enddo

    print *, "FFTW3_3D"

    mat3d_1 = 1.0d0

    mat3d_2 = mat3d_1

    call fftw3_3d(mat3d_1,norb,norb,norb,1)
    call fftw3_3d(mat3d_1,norb,norb,norb,-1)

    mat3d_1 = mat3d_1/(norb)**3

    err = 0.0d0

    do iorb = 1, norb   
        do jorb = 1, norb
            do korb = 1, norb
                err = mat3d_2(iorb, jorb, korb) - mat3d_1(iorb, jorb, korb)
                if (cdabs(err) .gt. 1.0d-8) then
                    print *, iorb, jorb, korb, cdabs(err), mat3d_1(iorb, jorb, korb), mat3d_2(iorb, jorb, korb)
                endif
            enddo
        enddo
    enddo

    do iorb=1, norb
       do is = 1, ns
          temp=(/iorb,is/)
          call indexing(norb*ns,2,(/norb,ns/),1,ind,temp)
          print '(5i5, 5f12.6)', ind, temp
       enddo
    enddo






end program TestCommon


