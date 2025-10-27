av_cold int ff_h264_decode_init(AVCodecContext *avctx)

{

    H264Context *h = avctx->priv_data;

    int ret;



    ret = h264_init_context(avctx, h);

    if (ret < 0)

        return ret;



    memset(h->pps.scaling_matrix4, 16, 6 * 16 * sizeof(uint8_t));

    memset(h->pps.scaling_matrix8, 16, 2 * 64 * sizeof(uint8_t));



    /* set defaults */

    // s->decode_mb = ff_h263_decode_mb;

    if (!avctx->has_b_frames)

        h->low_delay = 1;



    ff_h264_decode_init_vlc();



    ff_init_cabac_states();



    if (avctx->codec_id == AV_CODEC_ID_H264) {

        if (avctx->ticks_per_frame == 1)

            h->avctx->framerate.num *= 2;

        avctx->ticks_per_frame = 2;

    }



    if (avctx->extradata_size > 0 && avctx->extradata) {

       ret = ff_h264_decode_extradata(h);

       if (ret < 0) {

           ff_h264_free_context(h);

           return ret;

       }

    }



    if (h->sps.bitstream_restriction_flag &&

        h->avctx->has_b_frames < h->sps.num_reorder_frames) {

        h->avctx->has_b_frames = h->sps.num_reorder_frames;

        h->low_delay           = 0;

    }



    avctx->internal->allocate_progress = 1;



    if (h->enable_er) {

        av_log(avctx, AV_LOG_WARNING,

               "Error resilience is enabled. It is unsafe and unsupported and may crash. "

               "Use it at your own risk\n");

    }



    return 0;

}
