int ff_dxva2_common_end_frame(AVCodecContext *avctx, AVFrame *frame,

                              const void *pp, unsigned pp_size,

                              const void *qm, unsigned qm_size,

                              int (*commit_bs_si)(AVCodecContext *,

                                                  DECODER_BUFFER_DESC *bs,

                                                  DECODER_BUFFER_DESC *slice))

{

    AVDXVAContext *ctx = DXVA_CONTEXT(avctx);

    unsigned               buffer_count = 0;

#if CONFIG_D3D11VA

    D3D11_VIDEO_DECODER_BUFFER_DESC buffer11[4];

#endif

#if CONFIG_DXVA2

    DXVA2_DecodeBufferDesc          buffer2[4];

#endif

    DECODER_BUFFER_DESC             *buffer = NULL, *buffer_slice = NULL;

    int result, runs = 0;

    HRESULT hr;

    unsigned type;



    do {

        ff_dxva2_lock(avctx);

#if CONFIG_D3D11VA

        if (ff_dxva2_is_d3d11(avctx))

            hr = ID3D11VideoContext_DecoderBeginFrame(D3D11VA_CONTEXT(ctx)->video_context, D3D11VA_CONTEXT(ctx)->decoder,

                                                      get_surface(frame),

                                                      0, NULL);

#endif

#if CONFIG_DXVA2

        if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

            hr = IDirectXVideoDecoder_BeginFrame(DXVA2_CONTEXT(ctx)->decoder,

                                                 get_surface(frame),

                                                 NULL);

#endif

        if (hr != E_PENDING || ++runs > 50)

            break;

        ff_dxva2_unlock(avctx);

        av_usleep(2000);

    } while(1);



    if (FAILED(hr)) {

        av_log(avctx, AV_LOG_ERROR, "Failed to begin frame: 0x%x\n", hr);

        ff_dxva2_unlock(avctx);

        return -1;

    }



#if CONFIG_D3D11VA

    if (ff_dxva2_is_d3d11(avctx)) {

        buffer = &buffer11[buffer_count];

        type = D3D11_VIDEO_DECODER_BUFFER_PICTURE_PARAMETERS;

    }

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        buffer = &buffer2[buffer_count];

        type = DXVA2_PictureParametersBufferType;

    }

#endif

    result = ff_dxva2_commit_buffer(avctx, ctx, buffer,

                                    type,

                                    pp, pp_size, 0);

    if (result) {

        av_log(avctx, AV_LOG_ERROR,

               "Failed to add picture parameter buffer\n");

        goto end;

    }

    buffer_count++;



    if (qm_size > 0) {

#if CONFIG_D3D11VA

        if (ff_dxva2_is_d3d11(avctx)) {

            buffer = &buffer11[buffer_count];

            type = D3D11_VIDEO_DECODER_BUFFER_INVERSE_QUANTIZATION_MATRIX;

        }

#endif

#if CONFIG_DXVA2

        if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

            buffer = &buffer2[buffer_count];

            type = DXVA2_InverseQuantizationMatrixBufferType;

        }

#endif

        result = ff_dxva2_commit_buffer(avctx, ctx, buffer,

                                        type,

                                        qm, qm_size, 0);

        if (result) {

            av_log(avctx, AV_LOG_ERROR,

                   "Failed to add inverse quantization matrix buffer\n");

            goto end;

        }

        buffer_count++;

    }



#if CONFIG_D3D11VA

    if (ff_dxva2_is_d3d11(avctx)) {

        buffer       = &buffer11[buffer_count + 0];

        buffer_slice = &buffer11[buffer_count + 1];

    }

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        buffer       = &buffer2[buffer_count + 0];

        buffer_slice = &buffer2[buffer_count + 1];

    }

#endif



    result = commit_bs_si(avctx,

                          buffer,

                          buffer_slice);

    if (result) {

        av_log(avctx, AV_LOG_ERROR,

               "Failed to add bitstream or slice control buffer\n");

        goto end;

    }

    buffer_count += 2;



    /* TODO Film Grain when possible */



    assert(buffer_count == 1 + (qm_size > 0) + 2);



#if CONFIG_D3D11VA

    if (ff_dxva2_is_d3d11(avctx))

        hr = ID3D11VideoContext_SubmitDecoderBuffers(D3D11VA_CONTEXT(ctx)->video_context,

                                                     D3D11VA_CONTEXT(ctx)->decoder,

                                                     buffer_count, buffer11);

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        DXVA2_DecodeExecuteParams exec = {

            .NumCompBuffers     = buffer_count,

            .pCompressedBuffers = buffer2,

            .pExtensionData     = NULL,

        };

        hr = IDirectXVideoDecoder_Execute(DXVA2_CONTEXT(ctx)->decoder, &exec);

    }

#endif

    if (FAILED(hr)) {

        av_log(avctx, AV_LOG_ERROR, "Failed to execute: 0x%x\n", hr);

        result = -1;

    }



end:

#if CONFIG_D3D11VA

    if (ff_dxva2_is_d3d11(avctx))

        hr = ID3D11VideoContext_DecoderEndFrame(D3D11VA_CONTEXT(ctx)->video_context, D3D11VA_CONTEXT(ctx)->decoder);

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

        hr = IDirectXVideoDecoder_EndFrame(DXVA2_CONTEXT(ctx)->decoder, NULL);

#endif

    ff_dxva2_unlock(avctx);

    if (FAILED(hr)) {

        av_log(avctx, AV_LOG_ERROR, "Failed to end frame: 0x%x\n", hr);

        result = -1;

    }



    return result;

}
