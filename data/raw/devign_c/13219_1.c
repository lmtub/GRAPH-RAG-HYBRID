static av_cold int bfi_decode_init(AVCodecContext *avctx)

{

    BFIContext *bfi = avctx->priv_data;

    avctx->pix_fmt  = AV_PIX_FMT_PAL8;

    bfi->dst        = av_mallocz(avctx->width * avctx->height);



    return 0;

}