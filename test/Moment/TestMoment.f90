program TestMoment
  use Fourier
  use Common
  implicit none

  integer :: iomega, inu,ik, is, is1,is2,iorb1,iorb2, ii

  double precision :: beta, pi,omega_dble(0:9), nu_dble(0:9)
  complex*16 :: err, omega(0:9), nu(0:9), &
    glatt(4,4,2,5,0:9), glattref_moment(4,4,2,5,3),glattref_high(4,4,2,5), &
    flocdyn_moment(4,4,2,3), flocdyn_high(4,4,2), &
    flatdyn_moment(4,4,2,5,3), flatdyn_high(4,4,2,5), &

    wlatt(4,4,2,2,5,0:9), wlattref_moment(4,4,2,2,5,3),wlattref_high(4,4,2,2,5), &
    wref_moment_temp(4,4,2,2,3),wref_high_temp(4,4,2,2), &    
    blocdyn_moment(4,4,2,2,3), blocdyn_high(4,4,2,2), &
    blatdyn_moment(4,4,2,2,5,3), blatdyn_high(4,4,2,2,5)    

! glattm(4,4,2,5,3),&
!   gloc(4,4,2,0:9), glocmref(4,4,2,3), glocm(4,4,2,3), &
!   platt(4,4,2,2,5,0:9), plattmref(4,4,2,2,5,3), plattm(4,4,2,2,5,3), &
!   ploc(4,4,2,2,0:9), plocmref(4,4,2,2,3), plocm(4,4,2,2,3)      

  beta=1.0d0/(8.617333262145d-5*300.0d0)
  pi=datan2(1.0d0,1.0d0)*4.0d0

  omega=0.0d0

  do iomega=0, 9
    omega(iomega)=dcmplx(0.0d0, pi/beta*(2*iomega+1))
    omega_dble(iomega)=pi/beta*(2*iomega+1)
  enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  glatt=0.0d0
  glattref_moment=0.0d0
  glattref_high=0.0d0  


  do iomega=0, 9
    do ik=1, 5
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4
            if (iorb1 .eq. iorb2) then
              glattref_moment(iorb1,iorb2,is,ik,1)=1.0d0
            else
              glattref_moment(iorb1,iorb2,is,ik,1)=0.0d0
            endif
            glattref_moment(iorb1,iorb2,is,ik,2)=(ik*0.1d0+is*0.3d0+(iorb1+iorb2)/2.0d0)
            glattref_moment(iorb1,iorb2,is,ik,3)=(ik*0.01d0+is*0.23d0+(iorb1-iorb2)/2.0d0)
          enddo
        enddo
        glattref_moment(:,:,is,ik,2)=(glattref_moment(:,:,is,ik,2)+dconjg(transpose(glattref_moment(:,:,is,ik,2))))/2.0d0
        glattref_moment(:,:,is,ik,3)=(glattref_moment(:,:,is,ik,3)+dconjg(transpose(glattref_moment(:,:,is,ik,3))))/2.0d0
        do iorb1=1, 4
          do iorb2=1, 4
            glatt(iorb1,iorb2,is,ik,iomega)=glattref_moment(iorb1,iorb2,is,ik,1)/omega(iomega)+glattref_moment(iorb1,iorb2,is,ik,2)/(omega(iomega))**2+glattref_moment(iorb1,iorb2,is,ik,3)/omega(iomega)**3
          enddo
        enddo
      enddo
    enddo
  enddo

  call FLocDyn_M(4,2,10,omega_dble,glatt(:,:,:,1,:),1,1,flocdyn_moment,flocdyn_high)

  call FLatDyn_M(4,2,5,10,omega_dble,glatt,1,1,flatdyn_moment,flatdyn_high)  


  do is=1, 2
    do iorb1=1, 4
      do iorb2=1, 4
        do ii=1, 3          
          err=flocdyn_moment(iorb1,iorb2,is,ii)-glattref_moment(iorb1,iorb2,is,1,ii)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x,f12.6))',  iorb1,iorb2,is,ii, cdabs(err), flocdyn_moment(iorb1,iorb2,is,ii), glattref_moment(iorb1,iorb2,is,1,ii)
          endif
        end do
      enddo
    end do
  enddo



  do ik=1, 5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=flatdyn_moment(iorb1,iorb2,is,ik,ii)-glattref_moment(iorb1,iorb2,is,ik,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is,ik,ii,cdabs(err), flatdyn_moment(iorb1,iorb2,is,ik,ii), glattref_moment(iorb1,iorb2,is,ik,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  glatt=0.0d0
  glattref_moment=0.0d0
  glattref_high=0.0d0  


  do iomega=0, 9
    do ik=1, 5
      do is=1, 2
        do iorb1=1, 4
          do iorb2=1, 4
! if (iorb1 .eq. iorb2) then
!   glattref_moment(iorb1,iorb2,is,ik,1)=1.0d0
! else
!   glattref_moment(iorb1,iorb2,is,ik,1)=0.0d0
! endif
            glattref_moment(iorb1,iorb2,is,ik,1)=(ik*0.1d0+is*0.3d0+(iorb1+iorb2)/2.0d0)
            glattref_moment(iorb1,iorb2,is,ik,2)=(ik*0.01d0+is*0.23d0+(iorb1-iorb2)/2.0d0)
          enddo
        enddo
        glattref_moment(:,:,is,ik,1)=(glattref_moment(:,:,is,ik,1)+dconjg(transpose(glattref_moment(:,:,is,ik,1))))/2.0d0
        glattref_moment(:,:,is,ik,2)=(glattref_moment(:,:,is,ik,2)+dconjg(transpose(glattref_moment(:,:,is,ik,2))))/2.0d0
        do iorb1=1, 4
          do iorb2=1, 4
            glatt(iorb1,iorb2,is,ik,iomega)=glattref_moment(iorb1,iorb2,is,ik,1)/omega(iomega)+glattref_moment(iorb1,iorb2,is,ik,2)/(omega(iomega))**2+glattref_moment(iorb1,iorb2,is,ik,3)/omega(iomega)**3
          enddo
        enddo
      enddo
    enddo
  enddo

  call FLocDyn_M(4,2,10,omega_dble,glatt(:,:,:,1,:),0,1,flocdyn_moment,flocdyn_high)

  call FLatDyn_M(4,2,5,10,omega_dble,glatt,0,1,flatdyn_moment,flatdyn_high)  


  do is=1, 2
    do iorb1=1, 4
      do iorb2=1, 4
        do ii=1, 3          
          err=flocdyn_moment(iorb1,iorb2,is,ii)-glattref_moment(iorb1,iorb2,is,1,ii)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x,f12.6))',  iorb1,iorb2,is,ii, cdabs(err), flocdyn_moment(iorb1,iorb2,is,ii), glattref_moment(iorb1,iorb2,is,1,ii)
          endif
        end do
      enddo
    end do
  enddo



  do ik=1, 5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=flatdyn_moment(iorb1,iorb2,is,ik,ii)-glattref_moment(iorb1,iorb2,is,ik,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is,ik,ii,cdabs(err), flatdyn_moment(iorb1,iorb2,is,ik,ii), glattref_moment(iorb1,iorb2,is,ik,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  glatt=0.0d0
  glattref_moment=0.0d0
  glattref_high=0.0d0  



  do ik=1, 5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          if (iorb1 .eq. iorb2) then
            glattref_moment(iorb1,iorb2,is,ik,1)=1.0d0
          else
            glattref_moment(iorb1,iorb2,is,ik,1)=0.0d0
          endif
          glattref_moment(iorb1,iorb2,is,ik,2)=(ik*0.1d0+is*0.3d0+(iorb1+iorb2)/2.0d0)
          glattref_moment(iorb1,iorb2,is,ik,3)=(ik*0.01d0+is*0.23d0+(iorb1-iorb2)/2.0d0)
          glattref_high(iorb1,iorb2,is,ik)=(ik*0.3d0+is*0.5d0+(iorb1-iorb2)/8.0d0)          
        enddo
      enddo
      glattref_moment(:,:,is,ik,2)=(glattref_moment(:,:,is,ik,2)+dconjg(transpose(glattref_moment(:,:,is,ik,2))))/2.0d0
      glattref_moment(:,:,is,ik,3)=(glattref_moment(:,:,is,ik,3)+dconjg(transpose(glattref_moment(:,:,is,ik,3))))/2.0d0
      glattref_high(:,:,is,ik)=(glattref_high(:,:,is,ik)+dconjg(transpose(glattref_high(:,:,is,ik))))/2.0d0      
      do iorb1=1, 4
        do iorb2=1, 4
          do iomega=0, 9            
            glatt(iorb1,iorb2,is,ik,iomega)=glattref_moment(iorb1,iorb2,is,ik,1)/omega(iomega)+glattref_moment(iorb1,iorb2,is,ik,2)/(omega(iomega))**2+glattref_moment(iorb1,iorb2,is,ik,3)/omega(iomega)**3+glattref_high(iorb1,iorb2,is,ik)
          enddo
        enddo
      enddo
    enddo
  enddo

  call FLocDyn_M(4,2,10,omega_dble,glatt(:,:,:,1,:),0,0,flocdyn_moment,flocdyn_high)

  call FLatDyn_M(4,2,5,10,omega_dble,glatt,0,0,flatdyn_moment,flatdyn_high)  


  do is=1, 2
    do iorb1=1, 4
      do iorb2=1, 4
        do ii=1, 3          
          err=flocdyn_moment(iorb1,iorb2,is,ii)-glattref_moment(iorb1,iorb2,is,1,ii)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x,f12.6))',  iorb1,iorb2,is,ii, cdabs(err), flocdyn_moment(iorb1,iorb2,is,ii), glattref_moment(iorb1,iorb2,is,1,ii)
          endif
        end do
      enddo
    end do
  enddo



  do ik=1, 5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=flatdyn_moment(iorb1,iorb2,is,ik,ii)-glattref_moment(iorb1,iorb2,is,ik,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is,ik,ii,cdabs(err), flatdyn_moment(iorb1,iorb2,is,ik,ii), glattref_moment(iorb1,iorb2,is,ik,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo



  do is=1, 2
    do iorb1=1, 4
      do iorb2=1, 4
        err=flocdyn_high(iorb1,iorb2,is)-glattref_high(iorb1,iorb2,is,1)
        if (cdabs(err) .gt. 1.0d-6) then
          print '(3i5, 5(2x,f12.6))',  iorb1,iorb2,is, cdabs(err), flocdyn_high(iorb1,iorb2,is), glattref_high(iorb1,iorb2,is,1)
        endif
      enddo
    end do
  enddo



  do ik=1, 5
    do is=1, 2
      do iorb1=1, 4
        do iorb2=1, 4
          err=flatdyn_high(iorb1,iorb2,is,ik)-glattref_high(iorb1,iorb2,is,ik)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x,f12.6))', iorb1,iorb2,is,ik,cdabs(err), flatdyn_high(iorb1,iorb2,is,ik), glattref_high(iorb1,iorb2,is,ik)
          endif
        enddo
      end do
    enddo
  enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  nu=0.0d0

  do inu=0, 9
    nu(inu)=dcmplx(0.0d0, pi/beta*(2*inu))
    nu_dble(inu)=pi/beta*(2*inu)
  enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  wlatt=0.0d0
  wlattref_moment=0.0d0
  wlattref_high=0.0d0  



  do ik=1, 5
    wref_moment_temp=0.0d0
    wref_high_temp=0.0d0
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            if (iorb1 .eq. iorb2) then
              wref_moment_temp(iorb1,iorb2,is1,is2,1)=1.0d0
            else
              wref_moment_temp(iorb1,iorb2,is1,is2,1)=0.0d0
            endif
            wref_moment_temp(iorb1,iorb2,is1,is2,2)=(ik*0.1d0+is1*0.3d0+(iorb1+iorb2)/2.0d0)
            wref_moment_temp(iorb1,iorb2,is1,is2,3)=(ik*0.01d0+is1*0.23d0+(iorb1-iorb2)/2.0d0)
            wref_high_temp(iorb1,iorb2,is1,is2)=(ik*0.05d0+is1*0.8d0+(iorb1-iorb2)/3.0d0)            
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            wlattref_moment(iorb1,iorb2,is1,is2,ik,1)=(wref_moment_temp(iorb1,iorb2,is1,is2,1)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,1)))/2.0d0            
            wlattref_moment(iorb1,iorb2,is1,is2,ik,2)=(wref_moment_temp(iorb1,iorb2,is1,is2,2)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,2)))/2.0d0
            wlattref_moment(iorb1,iorb2,is1,is2,ik,3)=(wref_moment_temp(iorb1,iorb2,is1,is2,3)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,3)))/2.0d0
            wlattref_high(iorb1,iorb2,is1,is2,ik)=(wref_high_temp(iorb1,iorb2,is1,is2)+dconjg(wref_high_temp(iorb2,iorb1,is2,is1)))/2.0d0
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2            
        do inu=0, 9        
          do iorb1=1, 4
            do iorb2=1, 4
              wlatt(iorb1,iorb2,is1,is2,ik,inu)=wlattref_moment(iorb1,iorb2,is1,is2,ik,1)/nu(inu)+wlattref_moment(iorb1,iorb2,is1,is2,ik,2)/(nu(inu))**2+wlattref_moment(iorb1,iorb2,is1,is2,ik,3)/nu(inu)**3+wlattref_high(iorb1,iorb2,is1,is2,ik)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  call BLocDyn_M(4,2,10,nu_dble,wlatt(:,:,:,:,1,:),0,0,blocdyn_moment,blocdyn_high)

  call BLatDyn_M(4,2,5,10,nu_dble,wlatt,0,0,blatdyn_moment,blatdyn_high)  


  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=blocdyn_moment(iorb1,iorb2,is1,is2,ii)-wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))',  iorb1,iorb2,is1,is2,ii, cdabs(err), blocdyn_moment(iorb1,iorb2,is1,is2,ii), wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo



  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            do ii=1, 3          
              err=blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii)-wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(6i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,ik,ii,cdabs(err), blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii), wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              endif
            end do
          enddo
        end do
      enddo
    enddo
  enddo


  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          err=blocdyn_high(iorb1,iorb2,is1,is2)-wlattref_high(iorb1,iorb2,is1,is2,1)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x,f12.6))',  iorb1,iorb2,is1,is2, cdabs(err), blocdyn_high(iorb1,iorb2,is1,is2), wlattref_high(iorb1,iorb2,is1,is2,1)
          endif
        enddo
      end do
    enddo
  enddo



  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            err=blatdyn_high(iorb1,iorb2,is1,is2,ik)-wlattref_high(iorb1,iorb2,is1,is2,ik)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,ik,cdabs(err), blatdyn_high(iorb1,iorb2,is1,is2,ik), wlattref_high(iorb1,iorb2,is1,is2,ik)
            endif
          enddo
        end do
      enddo
    enddo
  enddo



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  wlatt=0.0d0
  wlattref_moment=0.0d0
  wlattref_high=0.0d0  



  do ik=1, 5
    wref_moment_temp=0.0d0
    wref_high_temp=0.0d0
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            if (iorb1 .eq. iorb2) then
              wref_moment_temp(iorb1,iorb2,is1,is2,1)=1.0d0
            else
              wref_moment_temp(iorb1,iorb2,is1,is2,1)=0.0d0
            endif
            wref_moment_temp(iorb1,iorb2,is1,is2,2)=(ik*0.1d0+is1*0.3d0+(iorb1+iorb2)/2.0d0)
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            wlattref_moment(iorb1,iorb2,is1,is2,ik,1)=(wref_moment_temp(iorb1,iorb2,is1,is2,1)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,1)))/2.0d0            
            wlattref_moment(iorb1,iorb2,is1,is2,ik,2)=(wref_moment_temp(iorb1,iorb2,is1,is2,2)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,2)))/2.0d0
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2            
        do inu=0, 9        
          do iorb1=1, 4
            do iorb2=1, 4
              wlatt(iorb1,iorb2,is1,is2,ik,inu)=wlattref_moment(iorb1,iorb2,is1,is2,ik,1)/nu(inu)+wlattref_moment(iorb1,iorb2,is1,is2,ik,2)/(nu(inu))**2
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  call BLocDyn_M(4,2,10,nu_dble,wlatt(:,:,:,:,1,:),0,1,blocdyn_moment,blocdyn_high)

  call BLatDyn_M(4,2,5,10,nu_dble,wlatt,0,1,blatdyn_moment,blatdyn_high)  


  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=blocdyn_moment(iorb1,iorb2,is1,is2,ii)-wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))',  iorb1,iorb2,is1,is2,ii, cdabs(err), blocdyn_moment(iorb1,iorb2,is1,is2,ii), wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo



  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            do ii=1, 3          
              err=blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii)-wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(6i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,ik,ii,cdabs(err), blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii), wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              endif
            end do
          enddo
        end do
      enddo
    enddo
  enddo

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  wlatt=0.0d0
  wlattref_moment=0.0d0
  wlattref_high=0.0d0  



  do ik=1, 5
    wref_moment_temp=0.0d0
    wref_high_temp=0.0d0
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            ! if (iorb1 .eq. iorb2) then
            !   wref_moment_temp(iorb1,iorb2,is1,is2,1)=1.0d0
            ! else
            !   wref_moment_temp(iorb1,iorb2,is1,is2,1)=0.0d0
            ! endif
            wref_moment_temp(iorb1,iorb2,is1,is2,2)=(ik*0.1d0+is1*0.3d0+(iorb1+iorb2)/2.0d0)
            ! wref_moment_temp(iorb1,iorb2,is1,is2,3)=(ik*0.01d0+is1*0.23d0+(iorb1-iorb2)/2.0d0)
            wref_high_temp(iorb1,iorb2,is1,is2)=(ik*0.05d0+is1*0.8d0+(iorb1-iorb2)/3.0d0)            
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            ! wlattref_moment(iorb1,iorb2,is1,is2,ik,1)=(wref_moment_temp(iorb1,iorb2,is1,is2,1)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,1)))/2.0d0            
            wlattref_moment(iorb1,iorb2,is1,is2,ik,2)=(wref_moment_temp(iorb1,iorb2,is1,is2,2)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,2)))/2.0d0
            ! wlattref_moment(iorb1,iorb2,is1,is2,ik,3)=(wref_moment_temp(iorb1,iorb2,is1,is2,3)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,3)))/2.0d0
            wlattref_high(iorb1,iorb2,is1,is2,ik)=(wref_high_temp(iorb1,iorb2,is1,is2)+dconjg(wref_high_temp(iorb2,iorb1,is2,is1)))/2.0d0
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2            
        do inu=0, 9        
          do iorb1=1, 4
            do iorb2=1, 4
              wlatt(iorb1,iorb2,is1,is2,ik,inu)=wlattref_moment(iorb1,iorb2,is1,is2,ik,2)/(nu(inu))**2+wlattref_high(iorb1,iorb2,is1,is2,ik)
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  call BLocDyn_M(4,2,10,nu_dble,wlatt(:,:,:,:,1,:),1,0,blocdyn_moment,blocdyn_high)

  call BLatDyn_M(4,2,5,10,nu_dble,wlatt,1,0,blatdyn_moment,blatdyn_high)  


  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=blocdyn_moment(iorb1,iorb2,is1,is2,ii)-wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))',  iorb1,iorb2,is1,is2,ii, cdabs(err), blocdyn_moment(iorb1,iorb2,is1,is2,ii), wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo



  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            do ii=1, 3          
              err=blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii)-wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(6i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,ik,ii,cdabs(err), blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii), wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              endif
            end do
          enddo
        end do
      enddo
    enddo
  enddo


  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          err=blocdyn_high(iorb1,iorb2,is1,is2)-wlattref_high(iorb1,iorb2,is1,is2,1)
          if (cdabs(err) .gt. 1.0d-6) then
            print '(4i5, 5(2x,f12.6))',  iorb1,iorb2,is1,is2, cdabs(err), blocdyn_high(iorb1,iorb2,is1,is2), wlattref_high(iorb1,iorb2,is1,is2,1)
          endif
        enddo
      end do
    enddo
  enddo



  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            err=blatdyn_high(iorb1,iorb2,is1,is2,ik)-wlattref_high(iorb1,iorb2,is1,is2,ik)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,ik,cdabs(err), blatdyn_high(iorb1,iorb2,is1,is2,ik), wlattref_high(iorb1,iorb2,is1,is2,ik)
            endif
          enddo
        end do
      enddo
    enddo
  enddo


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  wlatt=0.0d0
  wlattref_moment=0.0d0
  wlattref_high=0.0d0  



  do ik=1, 5
    wref_moment_temp=0.0d0
    wref_high_temp=0.0d0
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            ! if (iorb1 .eq. iorb2) then
            !   wref_moment_temp(iorb1,iorb2,is1,is2,1)=1.0d0
            ! else
            !   wref_moment_temp(iorb1,iorb2,is1,is2,1)=0.0d0
            ! endif
            wref_moment_temp(iorb1,iorb2,is1,is2,2)=(ik*0.1d0+is1*0.3d0+(iorb1+iorb2)/2.0d0)
            ! wref_moment_temp(iorb1,iorb2,is1,is2,3)=(ik*0.01d0+is1*0.23d0+(iorb1-iorb2)/2.0d0)
            ! wref_high_temp(iorb1,iorb2,is1,is2)=(ik*0.05d0+is1*0.8d0+(iorb1-iorb2)/3.0d0)            
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2        
        do iorb1=1, 4
          do iorb2=1, 4
            ! wlattref_moment(iorb1,iorb2,is1,is2,ik,1)=(wref_moment_temp(iorb1,iorb2,is1,is2,1)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,1)))/2.0d0            
            wlattref_moment(iorb1,iorb2,is1,is2,ik,2)=(wref_moment_temp(iorb1,iorb2,is1,is2,2)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,2)))/2.0d0
            ! wlattref_moment(iorb1,iorb2,is1,is2,ik,3)=(wref_moment_temp(iorb1,iorb2,is1,is2,3)+dconjg(wref_moment_temp(iorb2,iorb1,is2,is1,3)))/2.0d0
            ! wlattref_high(iorb1,iorb2,is1,is2,ik)=(wref_high_temp(iorb1,iorb2,is1,is2)+dconjg(wref_high_temp(iorb2,iorb1,is2,is1)))/2.0d0
          enddo
        enddo
      enddo
    enddo
    do is1=1, 2
      do is2=1, 2            
        do inu=0, 9        
          do iorb1=1, 4
            do iorb2=1, 4
              wlatt(iorb1,iorb2,is1,is2,ik,inu)=wlattref_moment(iorb1,iorb2,is1,is2,ik,2)/(nu(inu))**2
            enddo
          enddo
        enddo
      enddo
    enddo
  enddo

  call BLocDyn_M(4,2,10,nu_dble,wlatt(:,:,:,:,1,:),1,1,blocdyn_moment,blocdyn_high)

  call BLatDyn_M(4,2,5,10,nu_dble,wlatt,1,1,blatdyn_moment,blatdyn_high)  


  do is1=1, 2
    do is2=1, 2            
      do iorb1=1, 4
        do iorb2=1, 4
          do ii=1, 3          
            err=blocdyn_moment(iorb1,iorb2,is1,is2,ii)-wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            if (cdabs(err) .gt. 1.0d-6) then
              print '(5i5, 5(2x,f12.6))',  iorb1,iorb2,is1,is2,ii, cdabs(err), blocdyn_moment(iorb1,iorb2,is1,is2,ii), wlattref_moment(iorb1,iorb2,is1,is2,1,ii)
            endif
          end do
        enddo
      end do
    enddo
  enddo



  do ik=1, 5
    do is1=1, 2
      do is2=1, 2              
        do iorb1=1, 4
          do iorb2=1, 4
            do ii=1, 3          
              err=blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii)-wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              if (cdabs(err) .gt. 1.0d-6) then
                print '(6i5, 5(2x,f12.6))', iorb1,iorb2,is1,is2,ik,ii,cdabs(err), blatdyn_moment(iorb1,iorb2,is1,is2,ik,ii), wlattref_moment(iorb1,iorb2,is1,is2,ik,ii)
              endif
            end do
          enddo
        end do
      enddo
    enddo
  enddo



end program TestMoment
