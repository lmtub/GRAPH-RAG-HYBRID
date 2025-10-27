static void test_sum_square(void)

{

    INTFLOAT res0;

    INTFLOAT res1;

    LOCAL_ALIGNED_16(INTFLOAT, src, [256], [2]);



    declare_func(INTFLOAT, INTFLOAT (*x)[2], int n);



    randomize((INTFLOAT *)src, 256 * 2);

    res0 = call_ref(src, 256);

    res1 = call_new(src, 256);

    if (!float_near_abs_eps(res0, res1, EPS))

        fail();

    bench_new(src, 256);

}
