static av_cold int avs_decode_init(AVCodecContext * avctx)

{

    AvsContext *s = avctx->priv_data;



    s->frame = av_frame_alloc();

    if (!s->frame)

        return AVERROR(ENOMEM);



    avctx->pix_fmt = AV_PIX_FMT_PAL8;

    ff_set_dimensions(avctx, 318, 198);



    return 0;

}
