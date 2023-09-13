program TestDyson
  use Dyson
  use Common
  implicit none

  integer :: iomega,iorb1,iorb2,ik,is1,is2,inu,is
  integer*8 :: ind(2), nn1(2), nn2(2), ind1,ind2
  complex*16 :: omega(0:9), nu(0:9), tempmat1(4,4), tempmat2(4,4),err, &
    glatt0(4,4,2,5,0:9), glattref(4,4,2,5,0:9), siglatt(4,4,2,5,0:9), &
    glocstc(4,4,2), glatstc(4,4,2,5), glocdyn(4,4,2,0:9), glatdyn(4,4,2,5,0:9), &
    tempmat3(8,8),tempmat4(8,8),tempmat5(8,8), tempmat6(8,8),&
    wlatt0(4,4,2,2,5,0:9), wlattref(4,4,2,2,5,0:9), platt(4,4,2,2,5,0:9), &
    wlocstc(4,4,2,2), wlatstc(4,4,2,2,5), wlocdyn(4,4,2,2,0:9), wlatdyn(4,4,2,2,5,0:9)  

  do iomega=0, 9
    omega(iomega)=dcmplx(0.0d0,(2*iomega+1))
  enddo

  do iomega=0,9
    do ik=1, 5
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4
            if (iorb1 .eq. iorb2) then
              glatt0(iorb1,iorb2,is,ik,iomega)=1.0d0/(omega(iomega)-1.0d0+ik+is*0.1d0+(iorb1+iorb2)*2.0d0)
            end if
          enddo
        enddo
      enddo
    enddo
  enddo

! glatt0 test

  siglatt=0.0d0
  do iomega=0, 9
    do ik=1, 5    
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4
            siglatt(iorb1,iorb2,is,ik,iomega)=5*ik+iorb1+0.1d0*iomega+iorb2*2+iorb2
          enddo
        enddo

        call dcmplx_matinv(glatt0(:,:,is,ik,iomega), tempmat1, 4,4)
        tempmat2=tempmat1-siglatt(:,:,is,ik,iomega)
        call dcmplx_matinv(tempmat2, glattref(:,:,is,ik,iomega), 4,4)        
      enddo
    enddo
  enddo

  call FLocStc(4,2,glatt0(:,:,:,1,0), siglatt(:,:,:,1,0),glocstc)
  call FLatStc(4,2,5,glatt0(:,:,:,:,0), siglatt(:,:,:,:,0),glatstc)
  call FLocDyn(4,2,10,glatt0(:,:,:,1,:), siglatt(:,:,:,1,:),glocdyn)
  call FLatDyn(4,2,5,10,glatt0, siglatt,glatdyn)



  do is=1, 2
    do iorb1=1, 4
      do iorb2=1, 4
        err=glattref(iorb1,iorb2,is,1,0)-glocstc(iorb1,iorb2,is)
        if (cdabs(err) .gt. 1.0d-6) then
          print '(3i5, 5f12.6)', iorb1, iorb2, is, cdabs(err), glattref(iorb1,iorb2,is,1,0), glocstc(iorb1,iorb2,is)
        endif
      enddo
    enddo
  enddo

  do ik=1, 5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          err=glattref(iorb1,iorb2,is,ik,0)-glatstc(iorb1,iorb2,is,ik)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5f12.6)', iorb1, iorb2, is, ik,cdabs(err), glattref(iorb1,iorb2,is,ik,0), glatstc(iorb1,iorb2,is,ik)
          endif
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          err=glattref(iorb1,iorb2,is,1,iomega)-glocdyn(iorb1,iorb2,is,iomega)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5f12.6)', iorb1, iorb2, is, iomega,cdabs(err), glattref(iorb1,iorb2,is,1,iomega), glocdyn(iorb1,iorb2,is,iomega)
          endif
        enddo
      enddo
    enddo
  enddo


  do iomega=0, 9
    do ik=1, 5
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4
            err=glattref(iorb1,iorb2,is,ik,iomega)-glatdyn(iorb1,iorb2,is,ik,iomega)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5f12.6)', iorb1, iorb2, is, ik,iomega,cdabs(err), glattref(iorb1,iorb2,is,ik,iomega), glatdyn(iorb1,iorb2,is,ik,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo



  do inu=0, 9
    nu(inu)=dcmplx(0.0d0, (2*inu))
  enddo

  do inu=0,9
    do ik=1, 5
      do is1=1, 2
        do is2=1, 2        
          do iorb1=1, 4
            do iorb2=1, 4
              if (iorb1 .eq. iorb2) then
                wlatt0(iorb1,iorb2,is1,is2,ik,inu)=1.0d0/(1.0d0+ik+(is1-is2)*0.1d0+(iorb1+iorb2)*2.0d0)
              end if
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

! wlatt0 test

  platt=0.0d0
  do inu=0, 9
    do ik=1, 5    
      do is1=1, 2
        do is2=1, 2        
          do iorb1=1, 4
            do iorb2=1, 4
              platt(iorb1,iorb2,is1,is2,ik,inu)=5*ik+iorb1+0.1d0*inu+iorb2*2+iorb2+(is1-is2)*0.1d0
            enddo
          enddo
        enddo
      enddo

      tempmat3=0.0d0
      tempmat4=0.0d0
      do iorb1=1, 4
        do is1=1, 2
          nn1=(/iorb1,is1/)
          call indexing(8,2,(/4,2/),1,ind1,nn1)
          print *, ind1
          do iorb2=1, 4
            do is2=1, 2
              nn2=(/iorb2,is2/)
              call indexing(8,2,(/4,2/),1,ind2,nn2)
              tempmat3(ind1,ind2)=wlatt0(iorb1,iorb2,is1,is2,ik,inu)
              tempmat4(ind1,ind2)=platt(iorb1,iorb2,is1,is2,ik,inu)
            enddo
          enddo
        enddo
      enddo
      call dcmplx_matinv(tempmat3, tempmat5, 8,8)
      tempmat6=tempmat5-tempmat4
      call dcmplx_matinv(tempmat6, tempmat3, 8,8)

      do iorb1=1, 4
        do is1=1, 2
          nn1=(/iorb1,is1/)
          call indexing(8,2,(/4,2/),1,ind1,nn1)
          do iorb2=1, 4
            do is2=1, 2
              nn2=(/iorb2,is2/)
              call indexing(8,2,(/4,2/),1,ind2,nn2)
              wlattref(iorb1,iorb2,is1,is2,ik,inu)=tempmat3(ind1,ind2)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  call BLocStc(4,2,wlatt0(:,:,:,:,1,0), platt(:,:,:,:,1,0),wlocstc)
  call BLatStc(4,2,5,wlatt0(:,:,:,:,:,0), platt(:,:,:,:,:,0),wlatstc)
  call BLocDyn(4,2,10,wlatt0(:,:,:,:,1,:), platt(:,:,:,:,1,:),wlocdyn)
  call BLatDyn(4,2,5,10,wlatt0, platt,wlatdyn)


  print *, 'blocstc'

  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          err=wlattref(iorb1,iorb2,is1,is2,1,0)-wlocstc(iorb1,iorb2,is1,is2)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5f12.6)', iorb1, iorb2, is1,is2, cdabs(err), wlattref(iorb1,iorb2,is1,is2,1,0), wlocstc(iorb1,iorb2,is1,is2)
          endif
        enddo
      enddo
    enddo
  enddo
  print *, 'blatstc'  

  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            err=wlattref(iorb1,iorb2,is1,is2,ik,0)-wlatstc(iorb1,iorb2,is1,is2,ik)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5f12.6)', iorb1, iorb2, is1,is2, ik,cdabs(err), wlattref(iorb1,iorb2,is1,is2,ik,0), wlatstc(iorb1,iorb2,is1,is2,ik)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  print *, 'blocdyn'
  do inu=0, 9
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            err=wlattref(iorb1,iorb2,is1,is2,1,inu)-wlocdyn(iorb1,iorb2,is1,is2,inu)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5f12.6)', iorb1, iorb2, is1,is2, inu,cdabs(err), wlattref(iorb1,iorb2,is1,is2,1,inu), wlocdyn(iorb1,iorb2,is1,is2,inu)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo

  print *, 'blatdyn'
  do inu=0, 9
    do ik=1, 5
      do is1=1, 2
        do is2=1, 2                
          do iorb1=1, 4
            do iorb2=1, 4
              err=wlattref(iorb1,iorb2,is1,is2,ik,inu)-wlatdyn(iorb1,iorb2,is1,is2,ik,inu)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(6i5, 5f12.6)', iorb1, iorb2, is1,is2, ik,inu,cdabs(err), wlattref(iorb1,iorb2,is1,is2,ik,inu), wlatdyn(iorb1,iorb2,is1,is2,ik,inu)
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

end program TestDyson
