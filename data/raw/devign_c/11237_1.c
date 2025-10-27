static av_cold int dnxhd_decode_init(AVCodecContext *avctx)

{

    DNXHDContext *ctx = avctx->priv_data;



    ctx->avctx = avctx;

    ctx->cid = -1;

    avctx->colorspace = AVCOL_SPC_BT709;



    avctx->coded_width  = FFALIGN(avctx->width,  16);

    avctx->coded_height = FFALIGN(avctx->height, 16);



    ctx->rows = av_mallocz_array(avctx->thread_count, sizeof(RowContext));

    if (!ctx->rows)

        return AVERROR(ENOMEM);



    return 0;

}
