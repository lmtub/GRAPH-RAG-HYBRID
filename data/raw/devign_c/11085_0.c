static int test_vector_fmul(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                            const float *v1, const float *v2)

{

    LOCAL_ALIGNED(32, float, cdst, [LEN]);

    LOCAL_ALIGNED(32, float, odst, [LEN]);

    int ret;



    cdsp->vector_fmul(cdst, v1, v2, LEN);

    fdsp->vector_fmul(odst, v1, v2, LEN);



    if (ret = compare_floats(cdst, odst, LEN, FLT_EPSILON))

        av_log(NULL, AV_LOG_ERROR, "vector_fmul failed\n");



    return ret;

}
