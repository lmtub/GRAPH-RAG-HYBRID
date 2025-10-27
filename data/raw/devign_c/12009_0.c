int ff_nvenc_encode_frame(AVCodecContext *avctx, AVPacket *pkt,

                          const AVFrame *frame, int *got_packet)

{

    NVENCSTATUS nv_status;

    CUresult cu_res;

    CUcontext dummy;

    NvencSurface *tmpoutsurf, *inSurf;

    int res;



    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    NV_ENC_PIC_PARAMS pic_params = { 0 };

    pic_params.version = NV_ENC_PIC_PARAMS_VER;



    if (frame) {

        inSurf = get_free_frame(ctx);

        if (!inSurf) {

            av_log(avctx, AV_LOG_ERROR, "No free surfaces\n");

            return AVERROR_BUG;

        }



        cu_res = dl_fn->cuda_dl->cuCtxPushCurrent(ctx->cu_context);

        if (cu_res != CUDA_SUCCESS) {

            av_log(avctx, AV_LOG_ERROR, "cuCtxPushCurrent failed\n");

            return AVERROR_EXTERNAL;

        }



        res = nvenc_upload_frame(avctx, frame, inSurf);



        cu_res = dl_fn->cuda_dl->cuCtxPopCurrent(&dummy);

        if (cu_res != CUDA_SUCCESS) {

            av_log(avctx, AV_LOG_ERROR, "cuCtxPopCurrent failed\n");

            return AVERROR_EXTERNAL;

        }



        if (res) {

            inSurf->lockCount = 0;

            return res;

        }



        pic_params.inputBuffer = inSurf->input_surface;

        pic_params.bufferFmt = inSurf->format;

        pic_params.inputWidth = avctx->width;

        pic_params.inputHeight = avctx->height;

        pic_params.inputPitch = inSurf->pitch;

        pic_params.outputBitstream = inSurf->output_surface;



        if (avctx->flags & AV_CODEC_FLAG_INTERLACED_DCT) {

            if (frame->top_field_first)

                pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FIELD_TOP_BOTTOM;

            else

                pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FIELD_BOTTOM_TOP;

        } else {

            pic_params.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;

        }



        if (ctx->forced_idr >= 0 && frame->pict_type == AV_PICTURE_TYPE_I) {

            pic_params.encodePicFlags =

                ctx->forced_idr ? NV_ENC_PIC_FLAG_FORCEIDR : NV_ENC_PIC_FLAG_FORCEINTRA;

        } else {

            pic_params.encodePicFlags = 0;

        }



        pic_params.inputTimeStamp = frame->pts;



        nvenc_codec_specific_pic_params(avctx, &pic_params);

    } else {

        pic_params.encodePicFlags = NV_ENC_PIC_FLAG_EOS;

    }



    cu_res = dl_fn->cuda_dl->cuCtxPushCurrent(ctx->cu_context);

    if (cu_res != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "cuCtxPushCurrent failed\n");

        return AVERROR_EXTERNAL;

    }



    nv_status = p_nvenc->nvEncEncodePicture(ctx->nvencoder, &pic_params);



    cu_res = dl_fn->cuda_dl->cuCtxPopCurrent(&dummy);

    if (cu_res != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_ERROR, "cuCtxPopCurrent failed\n");

        return AVERROR_EXTERNAL;

    }



    if (nv_status != NV_ENC_SUCCESS &&

        nv_status != NV_ENC_ERR_NEED_MORE_INPUT)

        return nvenc_print_error(avctx, nv_status, "EncodePicture failed!");



    if (frame) {

        av_fifo_generic_write(ctx->output_surface_queue, &inSurf, sizeof(inSurf), NULL);

        timestamp_queue_enqueue(ctx->timestamp_list, frame->pts);



        if (ctx->initial_pts[0] == AV_NOPTS_VALUE)

            ctx->initial_pts[0] = frame->pts;

        else if (ctx->initial_pts[1] == AV_NOPTS_VALUE)

            ctx->initial_pts[1] = frame->pts;

    }



    /* all the pending buffers are now ready for output */

    if (nv_status == NV_ENC_SUCCESS) {

        while (av_fifo_size(ctx->output_surface_queue) > 0) {

            av_fifo_generic_read(ctx->output_surface_queue, &tmpoutsurf, sizeof(tmpoutsurf), NULL);

            av_fifo_generic_write(ctx->output_surface_ready_queue, &tmpoutsurf, sizeof(tmpoutsurf), NULL);

        }

    }



    if (output_ready(avctx, !frame)) {

        av_fifo_generic_read(ctx->output_surface_ready_queue, &tmpoutsurf, sizeof(tmpoutsurf), NULL);



        res = process_output_surface(avctx, pkt, tmpoutsurf);



        if (res)

            return res;



        av_assert0(tmpoutsurf->lockCount);

        tmpoutsurf->lockCount--;



        *got_packet = 1;

    } else {

        *got_packet = 0;

    }



    return 0;

}
