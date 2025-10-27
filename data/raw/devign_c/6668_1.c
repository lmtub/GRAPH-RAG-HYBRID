static av_cold int png_dec_init(AVCodecContext *avctx)

{

    PNGDecContext *s = avctx->priv_data;



    s->avctx = avctx;

    s->previous_picture.f = av_frame_alloc();

    s->last_picture.f = av_frame_alloc();

    s->picture.f = av_frame_alloc();

    if (!s->previous_picture.f || !s->last_picture.f || !s->picture.f)

        return AVERROR(ENOMEM);



    if (!avctx->internal->is_copy) {

        avctx->internal->allocate_progress = 1;

        ff_pngdsp_init(&s->dsp);

    }



    return 0;

}
