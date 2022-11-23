program TestBare
  use Bare
  use Common
  implicit none

  integer :: iomega,iorb,jorb,ik,is, ii
  double precision :: omega(0:9)
  complex*16 :: hmat(4,4,2,5), glatt(4,4,2,5,0:9), glattref(4,4,2,5,0:9), tempmat1(4,4), tempmat2(4,4), identity(4,4), err,gloc(4,4,2,0:9), ai

  ai=dcmplx(0.0d0, 1.0d0)
  do iomega=0, 9
    omega(iomega)=2*iomega+1
  enddo
  identity=0.0d0
  do ii=1, 4
    identity(ii,ii)=1.0d0
  enddo
  
  do ik=1, 5
    do is=1, 2
      do iorb=1, 4
        do jorb=1, 4
          if (iorb .eq. jorb) then
            hmat(iorb,jorb,is,ik)=1.0d0+ik+is*0.1d0+(iorb-jorb)*2.0d0
          else
            hmat(iorb,jorb,is,ik)=0.1d0+ik+is*0.1d0+(iorb-jorb)*0.1d0
          end if
        enddo
      enddo
    enddo
  enddo
  
  call FLocDyn(4,2,10,hmat(:,:,:,1),omega,gloc)

  call FLatDyn(4,2,5,10,hmat,omega,glatt)  
  
  do ik=1, 5  
    do is=1, 2
      do iomega=0, 9
        tempmat1=identity*omega(iomega)*ai-hmat(:,:,is,ik)
        call dcmplx_matinv(tempmat1, tempmat2,4,4)
        glattref(:,:,is,ik,iomega)=tempmat2
      enddo
    enddo
  enddo

  do iomega=0, 9
    do is=1, 2
      do iorb=1, 4
        do jorb=1, 4
          err=glattref(iorb,jorb,is,1,iomega)-gloc(iorb,jorb,is,iomega)
          if (cdabs(err) .gt. 1.0d-8) then
            print '(4i5, 5f12.6)', iorb, jorb, is, iomega, cdabs(err), glattref(iorb,jorb,is,ik,iomega), gloc(iorb,jorb,is,iomega)
          endif
        enddo
      enddo
    enddo
  enddo

  do ik=1, 5
    do iomega=0, 9
      do is=1, 2
        do iorb=1, 4
          do jorb=1, 4
            err=glattref(iorb,jorb,is,ik,iomega)-glatt(iorb,jorb,is,ik,iomega)
            if (cdabs(err) .gt. 1.0d-8) then
              print '(5i5, 5f12.6)', iorb, jorb, is,ik, iomega, cdabs(err), glattref(iorb,jorb,is,ik,iomega), glatt(iorb,jorb,is,ik,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo
      
end program TestBare
    
