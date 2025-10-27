static void vc1_sprite_flush(AVCodecContext *avctx)

{

    VC1Context *v     = avctx->priv_data;

    MpegEncContext *s = &v->s;

    AVFrame *f = &s->current_picture.f;

    int plane, i;



    /* Windows Media Image codecs have a convergence interval of two keyframes.

       Since we can't enforce it, clear to black the missing sprite. This is

       wrong but it looks better than doing nothing. */



    if (f->data[0])

        for (plane = 0; plane < (s->flags&CODEC_FLAG_GRAY ? 1 : 3); plane++)

            for (i = 0; i < v->sprite_height>>!!plane; i++)

                memset(f->data[plane] + i * f->linesize[plane],

                       plane ? 128 : 0, f->linesize[plane]);

}
