static int dvvideo_decode_frame(AVCodecContext *avctx,

                                 void *data, int *data_size,

                                 uint8_t *buf, int buf_size)

{

    DVVideoContext *s = avctx->priv_data;



    s->sys = dv_frame_profile(buf);

    if (!s->sys || buf_size < s->sys->frame_size)

        return -1; /* NOTE: we only accept several full frames */



    if(s->picture.data[0])

        avctx->release_buffer(avctx, &s->picture);



    s->picture.reference = 0;

    s->picture.key_frame = 1;

    s->picture.pict_type = FF_I_TYPE;

    avctx->pix_fmt = s->sys->pix_fmt;

    avcodec_set_dimensions(avctx, s->sys->width, s->sys->height);

    if(avctx->get_buffer(avctx, &s->picture) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }

    s->picture.interlaced_frame = 1;

    s->picture.top_field_first = 0;



    s->buf = buf;

    avctx->execute(avctx, dv_decode_mt, (void**)&dv_anchor[0], NULL,

                   s->sys->difseg_size * 27);



    emms_c();



    /* return image */

    *data_size = sizeof(AVFrame);

    *(AVFrame*)data= s->picture;



    return s->sys->frame_size;

}
