static av_cold int ipvideo_decode_init(AVCodecContext *avctx)

{

    IpvideoContext *s = avctx->priv_data;



    s->avctx = avctx;



    s->is_16bpp = avctx->bits_per_coded_sample == 16;

    avctx->pix_fmt = s->is_16bpp ? AV_PIX_FMT_RGB555 : AV_PIX_FMT_PAL8;



    ff_hpeldsp_init(&s->hdsp, avctx->flags);



    s->last_frame        = av_frame_alloc();

    s->second_last_frame = av_frame_alloc();

    s->cur_decode_frame  = av_frame_alloc();

    s->prev_decode_frame = av_frame_alloc();

    if (!s->last_frame || !s->second_last_frame ||

        !s->cur_decode_frame || !s->prev_decode_frame) {

        av_frame_free(&s->last_frame);

        av_frame_free(&s->second_last_frame);

        av_frame_free(&s->cur_decode_frame);

        av_frame_free(&s->prev_decode_frame);

        return AVERROR(ENOMEM);

    }



    s->cur_decode_frame->width   = avctx->width;

    s->prev_decode_frame->width  = avctx->width;

    s->cur_decode_frame->height  = avctx->height;

    s->prev_decode_frame->height = avctx->height;

    s->cur_decode_frame->format  = avctx->pix_fmt;

    s->prev_decode_frame->format = avctx->pix_fmt;



    ff_get_buffer(avctx, s->cur_decode_frame, 0);

    ff_get_buffer(avctx, s->prev_decode_frame, 0);



    return 0;

}
