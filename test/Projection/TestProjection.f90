program TestProjection
  use Projection
  implicit none


  integer :: iorb1,iorb2,is,ik,iomega,is1,is2
  complex*16 :: tempmat1(4,3), err, &
    glatt(4,4,2,5,0:9), gproj(4,3,2,5), glocref(3,3,2,5,0:9), &
    glocstc(3,3,2), glocdyn(3,3,2,0:9), glatstc(3,3,2,5), glatdyn(3,3,2,5,0:9), &
    wlatt(4,4,2,2,5,0:9), wproj(4,3,2,5), wlocref(3,3,2,2,5,0:9), &
    wlocstc(3,3,2,2), wlocdyn(3,3,2,2,0:9), wlatstc(3,3,2,2,5), wlatdyn(3,3,2,2,5,0:9)  


  glatt=0.0d0
  glocref=0.0d0
  gproj=0.0d0

  do iomega=0, 9
    do ik=1, 5
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4
            glatt(iorb1,iorb2,is,ik,iomega)=10*ik+iorb1+0.1d0*iomega+iorb2*2+dcmplx(0.0d0,(is)*0.1d0)
          enddo
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
        tempmat1=matmul(glatt(:,:,is,ik,iomega), gproj(:,:,is,ik))
        glocref(:,:,is,ik,iomega)=glocref(:,:,is,ik,iomega)+matmul(transpose(dconjg(gproj(:,:,is,ik))), tempmat1)
      enddo
    enddo
  enddo

  call FLocStc(4,2,glatt(:,:,:,1,0), 3,gproj(:,:,:,1),glocstc)
  call FLatStc(4,2,5,glatt(:,:,:,:,0), 3,gproj,glatstc)
  call FLocDyn(4,2,10,glatt(:,:,:,1,:), 3,gproj(:,:,:,1),glocdyn)
  call FLatDyn(4,2,5,10,glatt,3,gproj,glatdyn)


  do is=1, 2
    do iorb1=1, 3
      do iorb2=1, 3
        err=glocref(iorb1,iorb2,is,1,0)-glocstc(iorb1,iorb2,is)
        if (cdabs(err) .gt. 1.0d-6) then
          print '(3i5, 5(2x, f20.12))', iorb1, iorb2, is, cdabs(err), glocref(iorb1,iorb2,is,1,0), glocstc(iorb1,iorb2,is)
        endif
      enddo
    enddo
  enddo

  do ik=1, 5
    do is=1, 2
      do iorb1=1, 3
        do iorb2=1, 3      
          err=glocref(iorb1,iorb2,is,ik,0)-glatstc(iorb1,iorb2,is,ik)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is, ik,cdabs(err), glocref(iorb1,iorb2,is,ik,0), glatstc(iorb1,iorb2,is,ik)
          endif
        enddo
      enddo
    enddo
  enddo

  do iomega=0, 9
    do is=1, 2
      do iorb1=1, 3
        do iorb2=1, 3      
          err=glocref(iorb1,iorb2,is,1,iomega)-glocdyn(iorb1,iorb2,is,iomega)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x, f12.6))', iorb1, iorb2, is, iomega, cdabs(err), glocref(iorb1,iorb2,is,1,iomega), glocdyn(iorb1,iorb2,is,iomega)
          endif
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do ik=1, 5
      do is=1, 2
        do iorb1=1, 3
          do iorb2=1, 3      
            err=glocref(iorb1,iorb2,is,ik,iomega)-glatdyn(iorb1,iorb2,is,ik,iomega)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x, f12.6))', iorb1, iorb2, is, ik,iomega, cdabs(err), glocref(iorb1,iorb2,is,ik,iomega), glatdyn(iorb1,iorb2,is,ik,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo



  wlatt=0.0d0
  wlocref=0.0d0
  wproj=0.0d0

  do iomega=0, 9
    do ik=1, 5
      do is1=1, 2
        do is2=1, 2        
          do iorb1=1, 4
            do iorb2=1, 4
              wlatt(iorb1,iorb2,is1,is2,ik,iomega)=10*ik+iorb1+0.1d0*iomega+iorb2*2+dcmplx(0.0d0,(is1-is2)*0.1d0)
            enddo
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
          tempmat1=matmul(wlatt(:,:,is1,is2,ik,iomega), wproj(:,:,is2,ik))
          wlocref(:,:,is1,is2,ik,iomega)=wlocref(:,:,is1,is2,ik,iomega)+matmul(transpose(dconjg(wproj(:,:,is1,ik))), tempmat1)
        enddo
      enddo
    enddo
  enddo

  call BLocStc(4,2,wlatt(:,:,:,:,1,0), 3,wproj(:,:,:,1),wlocstc)
  call BLatStc(4,2,5,wlatt(:,:,:,:,:,0), 3,wproj,wlatstc)
  call BLocDyn(4,2,10,wlatt(:,:,:,:,1,:), 3,wproj(:,:,:,1),wlocdyn)
  call BLatDyn(4,2,5,10,wlatt,3,wproj,wlatdyn)

  print *, 'BLocStc'
  do is1=1, 2
    do is2=1, 2      
      do iorb1=1, 3
        do iorb2=1, 3
          err=wlocref(iorb1,iorb2,is1,is2,1,0)-wlocstc(iorb1,iorb2,is1,is2)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x, f20.12))', iorb1, iorb2, is1, is2,cdabs(err), wlocref(iorb1,iorb2,is1,is2,1,0), wlocstc(iorb1,iorb2,is1,is2)
          endif
        enddo
      enddo
    enddo
  enddo

  print *, 'BLatStc'  
  do ik=1, 5
    do is1=1, 2
      do is2=1, 2      
        do iorb1=1, 3
          do iorb2=1, 3      
            err=wlocref(iorb1,iorb2,is1,is2,ik,0)-wlatstc(iorb1,iorb2,is1,is2,ik)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x, f12.6))', iorb1, iorb2, is1, is2,ik,cdabs(err), wlocref(iorb1,iorb2,is1,is2,ik,0), wlatstc(iorb1,iorb2,is1,is2,ik)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  print *, 'BLocDyn'  
  do iomega=0, 9
    do is1=1, 2
      do is2=1, 2      
        do iorb1=1, 3
          do iorb2=1, 3      
            err=wlocref(iorb1,iorb2,is1,is2,1,iomega)-wlocdyn(iorb1,iorb2,is1,is2,iomega)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x, f12.6))', iorb1, iorb2, is1, is2,iomega, cdabs(err), wlocref(iorb1,iorb2,is1,is2,1,iomega), wlocdyn(iorb1,iorb2,is1,is2,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  print *, 'BLatDyn'  
  do iomega=0, 9
    do ik=1, 5
      do is1=1, 2
        do is2=1, 2        
          do iorb1=1, 3
            do iorb2=1, 3      
              err=wlocref(iorb1,iorb2,is1,is2,ik,iomega)-wlatdyn(iorb1,iorb2,is1,is2,ik,iomega)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(6i5, 5(2x, f12.6))', iorb1, iorb2, is1, is2,ik,iomega, cdabs(err), wlocref(iorb1,iorb2,is1,is2,ik,iomega), wlatdyn(iorb1,iorb2,is1,is2,ik,iomega)
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo
end program TestProjection

