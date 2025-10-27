static int dvvideo_decode_frame(AVCodecContext *avctx, void *data,

                                int *got_frame, AVPacket *avpkt)

{

    uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    DVVideoContext *s = avctx->priv_data;

    const uint8_t *vsc_pack;

    int apt, is16_9, ret;

    const AVDVProfile *sys;



    sys = av_dv_frame_profile(s->sys, buf, buf_size);

    if (!sys || buf_size < sys->frame_size) {

        av_log(avctx, AV_LOG_ERROR, "could not find dv frame profile\n");

        return -1; /* NOTE: we only accept several full frames */

    }



    if (sys != s->sys) {

        ret = ff_dv_init_dynamic_tables(s, sys);

        if (ret < 0) {

            av_log(avctx, AV_LOG_ERROR, "Error initializing the work tables.\n");

            return ret;

        }

        s->sys = sys;

    }



    s->frame            = data;

    s->frame->key_frame = 1;

    s->frame->pict_type = AV_PICTURE_TYPE_I;

    avctx->pix_fmt      = s->sys->pix_fmt;

    avctx->framerate    = av_inv_q(s->sys->time_base);



    ret = ff_set_dimensions(avctx, s->sys->width, s->sys->height);

    if (ret < 0)

        return ret;



    /* Determine the codec's sample_aspect ratio from the packet */

    vsc_pack = buf + 80 * 5 + 48 + 5;

    if (*vsc_pack == dv_video_control) {

        apt    = buf[4] & 0x07;

        is16_9 = (vsc_pack && ((vsc_pack[2] & 0x07) == 0x02 ||

                               (!apt && (vsc_pack[2] & 0x07) == 0x07)));

        ff_set_sar(avctx, s->sys->sar[is16_9]);

    }



    if (ff_get_buffer(avctx, s->frame, 0) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }

    s->frame->interlaced_frame = 1;

    s->frame->top_field_first  = 0;



    s->buf = buf;

    avctx->execute(avctx, dv_decode_video_segment, s->work_chunks, NULL,

                   dv_work_pool_size(s->sys), sizeof(DVwork_chunk));



    emms_c();



    /* return image */

    *got_frame = 1;



    return s->sys->frame_size;

}
