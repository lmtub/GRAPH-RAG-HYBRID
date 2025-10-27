static av_cold int encode_init_ls(AVCodecContext *ctx)

{

    ctx->coded_frame = av_frame_alloc();

    if (!ctx->coded_frame)

        return AVERROR(ENOMEM);



    ctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;

    ctx->coded_frame->key_frame = 1;



    if (ctx->pix_fmt != AV_PIX_FMT_GRAY8  &&

        ctx->pix_fmt != AV_PIX_FMT_GRAY16 &&

        ctx->pix_fmt != AV_PIX_FMT_RGB24  &&

        ctx->pix_fmt != AV_PIX_FMT_BGR24) {

        av_log(ctx, AV_LOG_ERROR,

               "Only grayscale and RGB24/BGR24 images are supported\n");

        return -1;

    }

    return 0;

}
