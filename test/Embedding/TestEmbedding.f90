program TestEmbedding
  use Embedding
  implicit none


  integer :: iorb1,iorb2,is,ik,iomega, is1,is2
  complex*16 :: tempmat1(4,3), err, &
    gloc(3,3,2,0:9), glattref(4,4,2,5,0:9), gproj(4,3,2,5), &
    glocstc(4,4,2), glatstc(4,4,2,5), glocdyn(4,4,2,0:9),glatdyn(4,4,2,5,0:9), &
    wloc(3,3,2,2,0:9), wlattref(4,4,2,2,5,0:9),wproj(4,3,2,5), &
    wlocstc(4,4,2,2), wlatstc(4,4,2,2,5), wlocdyn(4,4,2,2,0:9),wlatdyn(4,4,2,2,5,0:9), &    
    ! hlatt(3,3,2,5), hloc(2,2,2), hlattref(3,3,2,5), hproj(3,2,2,5), &

    ! vlatt(3,3,2,2,5), vloc(2,2,2,2), vlattref(3,3,2,2,5), vproj(3,2,2,5)    

! glatt(3,3,2,5,0:2), gloc(2,2,2,0:2), glocref(2,2,2,0:2), gproj(3,2,2,5),     

  glatt=0.0d0
  gloc=0.0d0
  glattref=0.0d0
  gproj=0.0d0

  do iomega=0, 9
    do is=1, 2
      do iorb1=1, 3
        do iorb2=1, 3
          gloc(iorb1,iorb2,is,iomega)=iorb1+0.1d0*iomega+iorb2*2+dcmplx(0.0d0,(is)*0.1d0)
        enddo
      enddo
    enddo
  enddo


  do ik=1, 5      
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 3
          gproj(iorb1,iorb2,is,ik)=0.1d0*is+ik*0.5d0+(iorb1-iorb2)*2.0d0
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do ik=1, 5      
      do is=1, 2
        tempmat1=matmul(gproj(:,:,is,ik), gloc(:,:,is,iomega))
        glattref(:,:,is,ik,iomega)=matmul(tempmat1, transpose(dconjg(gproj(:,:,is,ik))))
      enddo
    enddo
  enddo

  call FLocStc(4,2,gloc(:,:,:,0),3,gproj(:,:,:,1),glocstc)
  call FLatStc(4,2,5,gloc(:,:,:,0),3,gproj,glatstc)
  call FLocDyn(4,2,10,gloc,3,gproj(:,:,:,1),glocdyn)
  call FLatDyn(4,2,5,10,gloc,3,gproj,glatdyn)

  do is=1, 2
    do iorb1=1, 4
      do iorb2=1, 4      
        err=glattref(iorb1,iorb2,is,1,0)-glocstc(iorb1,iorb2,is)
        if (cdabs(err) .gt. 1.0d-6) then
          print '(3i5, 5(2x, f12.6))', iorb1, iorb2, is, cdabs(err), glattref(iorb1,iorb2,is,1,0), glocstc(iorb1,iorb2,is)
        endif
      enddo
    enddo
  enddo

  do iomega=0,9
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4      
          err=glattref(iorb1,iorb2,is,1,iomega)-glocdyn(iorb1,iorb2,is,iomega)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is, iomega,cdabs(err), glattref(iorb1,iorb2,is,1,iomega), glocdyn(iorb1,iorb2,is,iomega)
          endif
        enddo
      enddo
    enddo
  enddo


  do ik=1,5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          err=glattref(iorb1,iorb2,is,ik,0)-glatstc(iorb1,iorb2,is,ik)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is, ik,cdabs(err), glattref(iorb1,iorb2,is,ik,0), glatstc(iorb1,iorb2,is,ik)
          endif
        enddo
      enddo
    enddo
  enddo


  do iomega=0,9
    do ik=1,5
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4      
            err=glattref(iorb1,iorb2,is,ik,iomega)-glatdyn(iorb1,iorb2,is,ik,iomega)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x, f12.6))', iorb1, iorb2, is, ik,iomega,cdabs(err), glattref(iorb1,iorb2,is,ik,iomega), glatdyn(iorb1,iorb2,is,ik,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo



  wloc=0.0d0
  wlattref=0.0d0
  wproj=0.0d0

  do iomega=0, 9
    do is1=1, 2
      do is2=1, 2      
        do iorb1=1, 3
          do iorb2=1, 3
            wloc(iorb1,iorb2,is1,is2,iomega)=iorb1+0.1d0*iomega+iorb2*2+dcmplx(0.0d0,(is1-is2)*0.1d0)
          enddo
        enddo
      enddo
    enddo
  enddo


  do ik=1, 5      
    do is1=1, 2
      do iorb1=1, 4
        do iorb2=1, 3
          wproj(iorb1,iorb2,is1,ik)=0.1d0*is1+ik*0.5d0+(iorb1-iorb2)*2.0d0
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do ik=1, 5      
      do is1=1, 2
        do is2=1, 2        
          tempmat1=matmul(wproj(:,:,is1,ik), wloc(:,:,is1,is2,iomega))
          wlattref(:,:,is1,is2,ik,iomega)=matmul(tempmat1, transpose(dconjg(wproj(:,:,is2,ik))))
        enddo
      enddo
    enddo
  enddo
!!! from here
  call BLocStc(4,2,wloc(:,:,:,:,0),3,wproj(:,:,:,1),wlocstc)
  call BLatStc(4,2,5,wloc(:,:,:,:,0),3,wproj,wlatstc)
  call BLocDyn(4,2,10,wloc,3,wproj(:,:,:,1),wlocdyn)
  call BLatDyn(4,2,5,10,wloc,3,wproj,wlatdyn)  

  ! do iomega=0, 2
!   do ik=1, 5
  do is1=1, 2
    do is2=1, 2        
      do iorb1=1, 4
        do iorb2=1, 4      
          err=wlocstc(iorb1,iorb2,is1,is2)-wlattref(iorb1,iorb2,is1,is2,1,0)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is1,is2, cdabs(err), wlocstc(iorb1,iorb2,is1,is2), wlattref(iorb1,iorb2,is1,is2,1,0)
          endif
        enddo
      enddo
    enddo
  enddo

! do iomega=0, 2
  do ik=1, 5
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4      
            err=wlatstc(iorb1,iorb2,is1,is2,ik)-wlattref(iorb1,iorb2,is1,is2,ik,0)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is1,is2, cdabs(err), wlatstc(iorb1,iorb2,is1,is2,ik), wlattref(iorb1,iorb2,is1,is2,ik,0)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4      
            err=wlocdyn(iorb1,iorb2,is1,is2,iomega)-wlattref(iorb1,iorb2,is1,is2,1,iomega)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is1,is2, cdabs(err), wlocdyn(iorb1,iorb2,is1,is2,iomega), wlattref(iorb1,iorb2,is1,is2,1,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do ik=1, 5
      do is1=1, 2
        do is2=1, 2        
          do iorb1=1, 4
            do iorb2=1, 4      
              err=wlatdyn(iorb1,iorb2,is1,is2,ik,iomega)-wlattref(iorb1,iorb2,is1,is2,ik,iomega)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is1,is2, cdabs(err), wlatdyn(iorb1,iorb2,is1,is2,ik,iomega), wlattref(iorb1,iorb2,is1,is2,ik,iomega)
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
  


end program TestEmbedding

