void checkasm_check_jpeg2000dsp(void)

{

    LOCAL_ALIGNED_32(uint8_t, ref, [BUF_SIZE*3]);

    LOCAL_ALIGNED_32(uint8_t, new, [BUF_SIZE*3]);

    Jpeg2000DSPContext h;



    ff_jpeg2000dsp_init(&h);



    if (check_func(h.mct_decode[FF_DWT53], "jpeg2000_rct_int"))

        check_mct(&ref[BUF_SIZE*0], &ref[BUF_SIZE*1], &ref[BUF_SIZE*2],

                  &new[BUF_SIZE*0], &new[BUF_SIZE*1], &new[BUF_SIZE*2]);



    report("mct_decode");

}
