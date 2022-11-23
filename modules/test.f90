program test
  implicit none

  integer :: ii, jj, kk
  complex*16 :: aaa(10, 10, 10), bbb(10, 10, 10)


  do ii=1, 10
    do jj=1, 10
      do kk=1, 10
        aaa(kk, jj, ii)=(kk-1)*100+(jj-1)*10+ii-1
      enddo
    enddo
  enddo

  bbb=0.0d0
  call printaaa(aaa(1:3,2,1:4), bbb(1:3, 2, 1:4))
  
  do ii=1, 10
    do jj=1, 10
      print *, jj, ii, dble(aaa(jj, 2, ii)), dble(bbb(jj, 2, ii))
    enddo
  enddo

end program test


subroutine printaaa(aa, bb)
  implicit none
  complex*16, intent(in) :: aa(3, 4)
  complex*16, intent(out) :: bb(3,4)
  
  integer:: ii, jj

  bb=aa*-1

end subroutine printaaa
      
