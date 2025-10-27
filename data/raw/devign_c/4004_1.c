static av_cold int encode_close(AVCodecContext *avctx)

{

    if (avctx->priv_data) {

        DCAEncContext *c = avctx->priv_data;

        subband_bufer_free(c);

        ff_dcaadpcm_free(&c->adpcm_ctx);

    }

    return 0;

}
