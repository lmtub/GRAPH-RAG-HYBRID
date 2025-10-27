static av_cold int a64multi_close_encoder(AVCodecContext *avctx)

{

    A64Context *c = avctx->priv_data;

    av_frame_free(&avctx->coded_frame);

    av_free(c->mc_meta_charset);

    av_free(c->mc_best_cb);

    av_free(c->mc_charset);

    av_free(c->mc_charmap);

    av_free(c->mc_colram);

    return 0;

}
