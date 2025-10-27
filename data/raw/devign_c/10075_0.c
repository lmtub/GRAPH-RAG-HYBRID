static av_cold int common_init(AVCodecContext *avctx){

    FFV1Context *s = avctx->priv_data;



    s->avctx= avctx;

    s->flags= avctx->flags;



    avcodec_get_frame_defaults(&s->picture);



    ff_dsputil_init(&s->dsp, avctx);



    s->width = avctx->width;

    s->height= avctx->height;



    assert(s->width && s->height);

    //defaults

    s->num_h_slices=1;

    s->num_v_slices=1;





    return 0;

}
