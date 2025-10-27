static av_cold int libopenjpeg_encode_close(AVCodecContext *avctx)

{

    LibOpenJPEGContext *ctx = avctx->priv_data;



    opj_destroy_compress(ctx->compress);

    opj_image_destroy(ctx->image);

    av_freep(&avctx->coded_frame);

    return 0;

}
