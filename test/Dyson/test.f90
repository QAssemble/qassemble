program test
   use Common

   implicit none
   integer :: ind(2),iorb,is,irk

   do iorb = 1,4
      do is = 1,2
         ind = (/iorb,is/)
         print *, ind
         call indexing(8,2,(/4,2/),1,irk,ind)
         print *, irk
      enddo
   enddo

 
   print *, (/4,2/)

end program test
