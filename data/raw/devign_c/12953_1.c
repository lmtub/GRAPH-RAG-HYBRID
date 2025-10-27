static av_cold int ape_decode_close(AVCodecContext * avctx)

{

    APEContext *s = avctx->priv_data;

    int i;



    for (i = 0; i < APE_FILTER_LEVELS; i++)

        av_freep(&s->filterbuf[i]);



    av_freep(&s->data);

    return 0;

}
