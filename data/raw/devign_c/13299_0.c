void ff_vp3_idct_altivec(DCTELEM block[64])

{

    IDCT_START



    IDCT_1D(NOP, NOP)

    TRANSPOSE8(b0, b1, b2, b3, b4, b5, b6, b7);

    IDCT_1D(ADD8, SHIFT4)



    vec_st(b0, 0x00, block);

    vec_st(b1, 0x10, block);

    vec_st(b2, 0x20, block);

    vec_st(b3, 0x30, block);

    vec_st(b4, 0x40, block);

    vec_st(b5, 0x50, block);

    vec_st(b6, 0x60, block);

    vec_st(b7, 0x70, block);

}
