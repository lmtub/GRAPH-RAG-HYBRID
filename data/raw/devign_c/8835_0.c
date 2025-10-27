av_cold int ff_vaapi_encode_close(AVCodecContext *avctx)

{

    VAAPIEncodeContext *ctx = avctx->priv_data;

    VAAPIEncodePicture *pic, *next;



    for (pic = ctx->pic_start; pic; pic = next) {

        next = pic->next;

        vaapi_encode_free(avctx, pic);

    }



    if (ctx->va_context != VA_INVALID_ID) {

        vaDestroyContext(ctx->hwctx->display, ctx->va_context);

        ctx->va_context = VA_INVALID_ID;

    }



    if (ctx->va_config != VA_INVALID_ID) {

        vaDestroyConfig(ctx->hwctx->display, ctx->va_config);

        ctx->va_config = VA_INVALID_ID;

    }



    if (ctx->codec->close)

        ctx->codec->close(avctx);



    av_buffer_pool_uninit(&ctx->output_buffer_pool);



    av_freep(&ctx->codec_sequence_params);

    av_freep(&ctx->codec_picture_params);



    av_buffer_unref(&ctx->recon_frames_ref);

    av_buffer_unref(&ctx->input_frames_ref);

    av_buffer_unref(&ctx->device_ref);



    av_freep(&ctx->priv_data);



    return 0;

}
