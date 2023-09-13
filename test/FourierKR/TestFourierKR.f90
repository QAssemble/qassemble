program TestFourierKR
  use Fourier
  use Common
  implicit none

  integer :: iorb1, iorb2, is, ik, iomega, ik1, ik2, ik3, is1,is2
  integer*8 :: irk, ind(3) 
  complex*16 :: err,&
    fr(3,3,2,125,0:9), fk(3,3,2,125,0:9), fk2(3,3,2,125,0:9), &
  br(3,3,2,2,125,0:9), bk(3,3,2,2,125,0:9), bk2(3,3,2,2,125,0:9)  

  fr=0.0d0
  fk=0.0d0
  fk2=0.0d0    
  irk = 0
  ind = 0
  print *, "Fermion Lattice Dynamic test"
  do iomega=0, 9
    do ik3=0, 4
     do ik2=0, 4
       do ik1=0, 4          
         ind=(/ik1+1,ik2+1,ik3+1/)
         call indexing(125,3,(/5,5,5/),1,irk,ind)
!          if (iomega .eq. 0) then
!            print *, ind, irk
!          endif
         do is=1, 2
           do iorb1=1, 3
             do iorb2=1, 3
               fk(iorb1,iorb2,is,irk,iomega)=(iorb1-iorb2)/2.0d0+is*0.1+(ik1+ik2+ik3)/2.0+iomega*0.001d0
             enddo
           enddo
         enddo
       enddo
     enddo
   enddo
  enddo



  call FLatDyn_K2R(3,2,125,10,(/5,5,5/),fk,fr)
  call FLatDyn_R2K(3,2,125,10,(/5,5,5/),fr,fk2)

  do iomega=0, 9
    do irk=1, 125
      do is=1, 2
        do iorb1=1, 3
          do iorb2=1, 3
            err=fk(iorb1,iorb2,is,irk,iomega)-fk2(iorb1,iorb2,is,irk,iomega)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is,irk,iomega,cdabs(err),fk(iorb1,iorb2,is,irk,iomega), fk2(iorb1,iorb2,is,irk,iomega)
            endif
          enddo
        enddo
      enddo
    enddo
  enddo



  br=0.0d0
  bk=0.0d0
  bk2=0.0d0    
  print *, "Boson Lattice Dynamic test"
  do iomega=0, 9
    do ik3=0, 4
      do ik2=0, 4
        do ik1=0, 4          
          ind=(/ik1+1,ik2+1,ik3+1/)
          call indexing(125,3,(/5,5,5/),1,irk,ind)
!           if (iomega .eq. 0) then
!             print *, ind, irk
!           endif
          do is1=1, 2
            do is2=1, 2            
              do iorb1=1, 3
                do iorb2=1, 3
                  bk(iorb1,iorb2,is1,is2,irk,iomega)=(iorb1-iorb2)/2.0d0+(is1-is2)*0.1+(ik1+ik2+ik3)/2.0+iomega*0.001d0
                enddo
              enddo
            enddo
          end do
        enddo
      enddo
    enddo
  enddo



  call BLatDyn_K2R(3,2,125,10,(/5,5,5/),bk,br)
  
  call BLatDyn_R2K(3,2,125,10,(/5,5,5/),br,bk2)


  do iomega=0, 9
    do irk=1, 125
      do is1=1, 2
        do is2=1, 2        
          do iorb1=1, 3
            do iorb2=1, 3
              err=bk(iorb1,iorb2,is1,is2,irk,iomega)-bk2(iorb1,iorb2,is1,is2,irk,iomega)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,irk,iomega,cdabs(err),bk(iorb1,iorb2,is1,is2,irk,iomega), bk2(iorb1,iorb2,is1,is2,irk,iomega)
              endif
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo


end program TestFourierKR
