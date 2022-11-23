program test_modulo
  implicit none
  double precision:: tau, taumod, machep
  integer :: unitnum


  machep = epsilon ( machep )  

  tau=-3.5d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum  
  
  tau=-3.0d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum    

  tau=-2.0d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum    

  tau=0.0d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum


  tau=2.0d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum    


  tau=3.0d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum    


  tau=3.5d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum    

  tau=6.0d0
  taumod=modulo(tau, 3.0d0)
  unitnum=nint(tau-taumod)/3.0d0
  if (taumod .lt. machep) then
    unitnum=unitnum-1
  endif
  print '(3f12.6, i5)', tau, taumod, tau-3.0*unitnum, unitnum    


end program test_modulo
  
