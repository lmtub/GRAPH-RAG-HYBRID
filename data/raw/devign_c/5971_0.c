static void test_butterflies_float(const float *src0, const float *src1)

{

    LOCAL_ALIGNED_16(float,  cdst,  [LEN]);

    LOCAL_ALIGNED_16(float,  odst,  [LEN]);

    LOCAL_ALIGNED_16(float,  cdst1, [LEN]);

    LOCAL_ALIGNED_16(float,  odst1, [LEN]);

    int i;



    declare_func(void, float *av_restrict src0, float *av_restrict src1,

    int len);



    memcpy(cdst,  src0, LEN * sizeof(*src0));

    memcpy(cdst1, src1, LEN * sizeof(*src1));

    memcpy(odst,  src0, LEN * sizeof(*src0));

    memcpy(odst1, src1, LEN * sizeof(*src1));



    call_ref(cdst, cdst1, LEN);

    call_new(odst, odst1, LEN);

    for (i = 0; i < LEN; i++) {

        if (!float_near_abs_eps(cdst[i], odst[i], FLT_EPSILON)) {

            fprintf(stderr, "%d: %- .12f - %- .12f = % .12g\n",

                    i, cdst[i], odst[i], cdst[i] - odst[i]);

            fail();

            break;

        }

    }

    memcpy(odst,  src0, LEN * sizeof(*src0));

    memcpy(odst1, src1, LEN * sizeof(*src1));

    bench_new(odst, odst1, LEN);

}
