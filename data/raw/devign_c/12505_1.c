static void wavpack_decode_flush(AVCodecContext *avctx)

{

    WavpackContext *s = avctx->priv_data;

    int i;



    for (i = 0; i < s->fdec_num; i++)

        wv_reset_saved_context(s->fdec[i]);

}
