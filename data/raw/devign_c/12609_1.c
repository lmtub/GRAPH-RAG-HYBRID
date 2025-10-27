X264_close(AVCodecContext *avctx)
{
    X264Context *x4 = avctx->priv_data;
    if(x4->enc)
        x264_encoder_close(x4->enc);
    return 0;
}