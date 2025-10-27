static av_cold int nvenc_open_session(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encode_session_params = { 0 };

    NVENCSTATUS nv_status;



    encode_session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;

    encode_session_params.apiVersion = NVENCAPI_VERSION;

    encode_session_params.device = ctx->cu_context;

    encode_session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;



    nv_status = p_nvenc->nvEncOpenEncodeSessionEx(&encode_session_params, &ctx->nvencoder);

    if (nv_status != NV_ENC_SUCCESS) {

        ctx->nvencoder = NULL;

        return nvenc_print_error(avctx, nv_status, "OpenEncodeSessionEx failed");

    }



    return 0;

}
