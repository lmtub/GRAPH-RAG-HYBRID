static av_cold int cinepak_encode_end(AVCodecContext *avctx)

{

    CinepakEncContext *s = avctx->priv_data;

    int x;



    av_free(s->codebook_input);

    av_free(s->codebook_closest);

    av_free(s->strip_buf);

    av_free(s->frame_buf);

    av_free(s->mb);

#ifdef CINEPAKENC_DEBUG

    av_free(s->best_mb);

#endif



    for(x = 0; x < 3; x++)

        av_free(s->pict_bufs[x]);



    av_log(avctx, AV_LOG_INFO, "strip coding stats: %i V1 mode, %i V4 mode, %i MC mode (%i V1 encs, %i V4 encs, %i skips)\n",

        s->num_v1_mode, s->num_v4_mode, s->num_mc_mode, s->num_v1_encs, s->num_v4_encs, s->num_skips);



    return 0;

}
