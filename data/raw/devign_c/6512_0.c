static int test_vector_fmac_scalar(AVFloatDSPContext *fdsp, AVFloatDSPContext *cdsp,

                                   const float *v1, const float *src0, float scale)

{

    LOCAL_ALIGNED(32, float, cdst, [LEN]);

    LOCAL_ALIGNED(32, float, odst, [LEN]);

    int ret;



    memcpy(cdst, v1, LEN * sizeof(*v1));

    memcpy(odst, v1, LEN * sizeof(*v1));



    cdsp->vector_fmac_scalar(cdst, src0, scale, LEN);

    fdsp->vector_fmac_scalar(odst, src0, scale, LEN);



    if (ret = compare_floats(cdst, odst, LEN, ARBITRARY_FMAC_SCALAR_CONST))

        av_log(NULL, AV_LOG_ERROR, "vector_fmac_scalar failed\n");



    return ret;

}
