long check_dcbzl_effect(void)

{

  register char *fakedata = (char*)av_malloc(1024);

  register char *fakedata_middle;

  register long zero = 0;

  register long i = 0;

  long count = 0;



  if (!fakedata)

  {

    return 0L;

  }



  fakedata_middle = (fakedata + 512);



  memset(fakedata, 0xFF, 1024);



  /* below the constraint "b" seems to mean "Address base register"

     in gcc-3.3 / RS/6000 speaks. seems to avoid using r0, so.... */

  asm volatile("dcbzl %0, %1" : : "b" (fakedata_middle), "r" (zero));



  for (i = 0; i < 1024 ; i ++)

  {

    if (fakedata[i] == (char)0)

      count++;

  }



  av_free(fakedata);



  return count;

}
