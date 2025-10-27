static av_cold int oggvorbis_encode_close(AVCodecContext *avctx)

{

    OggVorbisContext *s = avctx->priv_data;



    /* notify vorbisenc this is EOF */

    vorbis_analysis_wrote(&s->vd, 0);



    vorbis_block_clear(&s->vb);

    vorbis_dsp_clear(&s->vd);

    vorbis_info_clear(&s->vi);



    av_freep(&avctx->coded_frame);

    av_freep(&avctx->extradata);



    return 0;

}
