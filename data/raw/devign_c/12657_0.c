static int test_vector_dmul_scalar(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                                   const double *v1, double scale)

{

    LOCAL_ALIGNED(32, double, cdst, [LEN]);

    LOCAL_ALIGNED(32, double, odst, [LEN]);

    int ret;



    cdsp->vector_dmul_scalar(cdst, v1, scale, LEN);

    fdsp->vector_dmul_scalar(odst, v1, scale, LEN);



    if (ret = compare_doubles(cdst, odst, LEN, DBL_EPSILON))

        av_log(NULL, AV_LOG_ERROR, "vector_dmul_scalar failed\n");



    return ret;

}
