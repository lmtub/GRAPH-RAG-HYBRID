void checkasm_check_fixed_dsp(void)

{

    LOCAL_ALIGNED_32(int32_t, src0, [BUF_SIZE]);

    LOCAL_ALIGNED_32(int32_t, src1, [BUF_SIZE]);

    LOCAL_ALIGNED_32(int32_t, src2, [BUF_SIZE]);

    AVFixedDSPContext *fdsp = avpriv_alloc_fixed_dsp(1);



    randomize_buffers();

    if (check_func(fdsp->vector_fmul, "vector_fmul"))

        check_vector_fmul(src0, src1);

    if (check_func(fdsp->vector_fmul_add, "vector_fmul_add"))

        check_vector_fmul_add(src0, src1, src2);

    if (check_func(fdsp->vector_fmul_reverse, "vector_fmul_reverse"))

        check_vector_fmul(src0, src1);

    if (check_func(fdsp->vector_fmul_window, "vector_fmul_window"))

        check_vector_fmul_window(src0, src1, src2);

    if (check_func(fdsp->vector_fmul_window_scaled, "vector_fmul_window_scaled"))

        check_vector_fmul_window_scaled(src0, src1, src2);

    report("vector_fmul");

    if (check_func(fdsp->butterflies_fixed, "butterflies_fixed"))

        check_butterflies(src0, src1);

    report("butterflies_fixed");

    if (check_func(fdsp->scalarproduct_fixed, "scalarproduct_fixed"))

        check_scalarproduct_fixed(src0, src1);

    report("scalarproduct_fixed");



    av_freep(&fdsp);

}
