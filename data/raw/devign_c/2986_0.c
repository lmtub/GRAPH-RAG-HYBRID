static av_cold int indeo3_decode_init(AVCodecContext *avctx)

{

    Indeo3DecodeContext *s = avctx->priv_data;



    s->avctx = avctx;

    s->width = avctx->width;

    s->height = avctx->height;

    avctx->pix_fmt = PIX_FMT_YUV410P;



    build_modpred(s);

    iv_alloc_frames(s);



    return 0;

}
