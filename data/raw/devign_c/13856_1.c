static int nvenc_upload_frame(AVCodecContext *avctx, const AVFrame *frame,

                                      NvencSurface *nvenc_frame)

{

    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    int res;

    NVENCSTATUS nv_status;



    if (avctx->pix_fmt == AV_PIX_FMT_CUDA || avctx->pix_fmt == AV_PIX_FMT_D3D11) {

        int reg_idx = nvenc_register_frame(avctx, frame);

        if (reg_idx < 0) {

            av_log(avctx, AV_LOG_ERROR, "Could not register an input HW frame\n");

            return reg_idx;

        }



        res = av_frame_ref(nvenc_frame->in_ref, frame);

        if (res < 0)

            return res;



        nvenc_frame->in_map.version = NV_ENC_MAP_INPUT_RESOURCE_VER;

        nvenc_frame->in_map.registeredResource = ctx->registered_frames[reg_idx].regptr;

        nv_status = p_nvenc->nvEncMapInputResource(ctx->nvencoder, &nvenc_frame->in_map);

        if (nv_status != NV_ENC_SUCCESS) {

            av_frame_unref(nvenc_frame->in_ref);

            return nvenc_print_error(avctx, nv_status, "Error mapping an input resource");

        }



        ctx->registered_frames[reg_idx].mapped = 1;

        nvenc_frame->reg_idx                   = reg_idx;

        nvenc_frame->input_surface             = nvenc_frame->in_map.mappedResource;

        nvenc_frame->format                    = nvenc_frame->in_map.mappedBufferFmt;

        nvenc_frame->pitch                     = frame->linesize[0];

        return 0;

    } else {

        NV_ENC_LOCK_INPUT_BUFFER lockBufferParams = { 0 };



        lockBufferParams.version = NV_ENC_LOCK_INPUT_BUFFER_VER;

        lockBufferParams.inputBuffer = nvenc_frame->input_surface;



        nv_status = p_nvenc->nvEncLockInputBuffer(ctx->nvencoder, &lockBufferParams);

        if (nv_status != NV_ENC_SUCCESS) {

            return nvenc_print_error(avctx, nv_status, "Failed locking nvenc input buffer");

        }



        nvenc_frame->pitch = lockBufferParams.pitch;

        res = nvenc_copy_frame(avctx, nvenc_frame, &lockBufferParams, frame);



        nv_status = p_nvenc->nvEncUnlockInputBuffer(ctx->nvencoder, nvenc_frame->input_surface);

        if (nv_status != NV_ENC_SUCCESS) {

            return nvenc_print_error(avctx, nv_status, "Failed unlocking input buffer!");

        }



        return res;

    }

}
