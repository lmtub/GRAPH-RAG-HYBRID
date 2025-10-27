static int test_vector_fmul_add(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                                const float *v1, const float *v2, const float *v3)

{

    LOCAL_ALIGNED(32, float, cdst, [LEN]);

    LOCAL_ALIGNED(32, float, odst, [LEN]);

    int ret;



    cdsp->vector_fmul_add(cdst, v1, v2, v3, LEN);

    fdsp->vector_fmul_add(odst, v1, v2, v3, LEN);



    if (ret = compare_floats(cdst, odst, LEN, ARBITRARY_FMUL_ADD_CONST))

        av_log(NULL, AV_LOG_ERROR, "vector_fmul_add failed\n");



    return ret;

}
