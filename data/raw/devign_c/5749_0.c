static int test_vector_fmul_scalar(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                                   const float *v1, float scale)

{

    LOCAL_ALIGNED(32, float, cdst, [LEN]);

    LOCAL_ALIGNED(32, float, odst, [LEN]);

    int ret;



    cdsp->vector_fmul_scalar(cdst, v1, scale, LEN);

    fdsp->vector_fmul_scalar(odst, v1, scale, LEN);



    if (ret = compare_floats(cdst, odst, LEN, FLT_EPSILON))

        av_log(NULL, AV_LOG_ERROR, "vector_fmul_scalar failed\n");



    return ret;

}
