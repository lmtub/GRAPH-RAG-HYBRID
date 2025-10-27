static av_always_inline void FUNC(row_fdct)(int16_t *data)

{

  int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;

  int tmp10, tmp11, tmp12, tmp13;

  int z1, z2, z3, z4, z5;

  int16_t *dataptr;

  int ctr;



  /* Pass 1: process rows. */

  /* Note results are scaled up by sqrt(8) compared to a true DCT; */

  /* furthermore, we scale the results by 2**PASS1_BITS. */



  dataptr = data;

  for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {

    tmp0 = dataptr[0] + dataptr[7];

    tmp7 = dataptr[0] - dataptr[7];

    tmp1 = dataptr[1] + dataptr[6];

    tmp6 = dataptr[1] - dataptr[6];

    tmp2 = dataptr[2] + dataptr[5];

    tmp5 = dataptr[2] - dataptr[5];

    tmp3 = dataptr[3] + dataptr[4];

    tmp4 = dataptr[3] - dataptr[4];



    /* Even part per LL&M figure 1 --- note that published figure is faulty;

     * rotator "sqrt(2)*c1" should be "sqrt(2)*c6".

     */



    tmp10 = tmp0 + tmp3;

    tmp13 = tmp0 - tmp3;

    tmp11 = tmp1 + tmp2;

    tmp12 = tmp1 - tmp2;



    dataptr[0] = (int16_t) ((tmp10 + tmp11) << PASS1_BITS);

    dataptr[4] = (int16_t) ((tmp10 - tmp11) << PASS1_BITS);



    z1 = MULTIPLY(tmp12 + tmp13, FIX_0_541196100);

    dataptr[2] = (int16_t) DESCALE(z1 + MULTIPLY(tmp13, FIX_0_765366865),

                                   CONST_BITS-PASS1_BITS);

    dataptr[6] = (int16_t) DESCALE(z1 + MULTIPLY(tmp12, - FIX_1_847759065),

                                   CONST_BITS-PASS1_BITS);



    /* Odd part per figure 8 --- note paper omits factor of sqrt(2).

     * cK represents cos(K*pi/16).

     * i0..i3 in the paper are tmp4..tmp7 here.

     */



    z1 = tmp4 + tmp7;

    z2 = tmp5 + tmp6;

    z3 = tmp4 + tmp6;

    z4 = tmp5 + tmp7;

    z5 = MULTIPLY(z3 + z4, FIX_1_175875602); /* sqrt(2) * c3 */



    tmp4 = MULTIPLY(tmp4, FIX_0_298631336); /* sqrt(2) * (-c1+c3+c5-c7) */

    tmp5 = MULTIPLY(tmp5, FIX_2_053119869); /* sqrt(2) * ( c1+c3-c5+c7) */

    tmp6 = MULTIPLY(tmp6, FIX_3_072711026); /* sqrt(2) * ( c1+c3+c5-c7) */

    tmp7 = MULTIPLY(tmp7, FIX_1_501321110); /* sqrt(2) * ( c1+c3-c5-c7) */

    z1 = MULTIPLY(z1, - FIX_0_899976223); /* sqrt(2) * (c7-c3) */

    z2 = MULTIPLY(z2, - FIX_2_562915447); /* sqrt(2) * (-c1-c3) */

    z3 = MULTIPLY(z3, - FIX_1_961570560); /* sqrt(2) * (-c3-c5) */

    z4 = MULTIPLY(z4, - FIX_0_390180644); /* sqrt(2) * (c5-c3) */



    z3 += z5;

    z4 += z5;



    dataptr[7] = (int16_t) DESCALE(tmp4 + z1 + z3, CONST_BITS-PASS1_BITS);

    dataptr[5] = (int16_t) DESCALE(tmp5 + z2 + z4, CONST_BITS-PASS1_BITS);

    dataptr[3] = (int16_t) DESCALE(tmp6 + z2 + z3, CONST_BITS-PASS1_BITS);

    dataptr[1] = (int16_t) DESCALE(tmp7 + z1 + z4, CONST_BITS-PASS1_BITS);



    dataptr += DCTSIZE;         /* advance pointer to next row */

  }

}
