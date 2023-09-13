program test

  implicit none

  integer :: iomega, ik1, ik2, ik3, ind(3)

  do iomega=0, 9
    do ik3=0, 4
      do ik2=0, 4
        do ik1=0, 4
          ind=(/ik1+1,ik2+1,ik3+1/)
          print *, ind
        enddo
      enddo
    enddo
  enddo


end program test
