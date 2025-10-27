static av_cold int png_dec_end(AVCodecContext *avctx)

{

    PNGDecContext *s = avctx->priv_data;



    ff_thread_release_buffer(avctx, &s->previous_picture);

    av_frame_free(&s->previous_picture.f);

    ff_thread_release_buffer(avctx, &s->last_picture);

    av_frame_free(&s->last_picture.f);

    ff_thread_release_buffer(avctx, &s->picture);

    av_frame_free(&s->picture.f);

    av_freep(&s->buffer);

    s->buffer_size = 0;

    av_freep(&s->last_row);

    s->last_row_size = 0;

    av_freep(&s->tmp_row);

    s->tmp_row_size = 0;

    av_freep(&s->extra_data);

    s->extra_data_size = 0;



    return 0;

}
