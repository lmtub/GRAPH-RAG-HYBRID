static void test_hybrid_analysis(void)

{

    LOCAL_ALIGNED_16(INTFLOAT, dst0, [BUF_SIZE], [2]);

    LOCAL_ALIGNED_16(INTFLOAT, dst1, [BUF_SIZE], [2]);

    LOCAL_ALIGNED_16(INTFLOAT, in, [12], [2]);

    LOCAL_ALIGNED_16(INTFLOAT, filter, [N], [8][2]);



    declare_func(void, INTFLOAT (*out)[2], INTFLOAT (*in)[2],

                 const INTFLOAT (*filter)[8][2],

                 ptrdiff_t stride, int n);



    randomize((INTFLOAT *)in, 12 * 2);

    randomize((INTFLOAT *)filter, N * 8 * 2);



    randomize((INTFLOAT *)dst0, BUF_SIZE * 2);

    memcpy(dst1, dst0, BUF_SIZE * 2 * sizeof(INTFLOAT));



    call_ref(dst0, in, filter, STRIDE, N);

    call_new(dst1, in, filter, STRIDE, N);



    if (!float_near_abs_eps_array((float *)dst0, (float *)dst1, EPS, BUF_SIZE * 2))

        fail();

    bench_new(dst1, in, filter, STRIDE, N);

}
