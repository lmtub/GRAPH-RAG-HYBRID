static int raw_init_encoder(AVCodecContext *avctx)

{

    avctx->coded_frame = (AVFrame *)avctx->priv_data;

    avctx->coded_frame->pict_type = FF_I_TYPE;

    avctx->coded_frame->key_frame = 1;

    avctx->codec_tag = findFourCC(avctx->pix_fmt);

    return 0;

}
