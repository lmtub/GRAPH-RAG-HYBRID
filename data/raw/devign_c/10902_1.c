int attribute_align_arg avcodec_decode_audio4(AVCodecContext *avctx,

                                              AVFrame *frame,

                                              int *got_frame_ptr,

                                              AVPacket *avpkt)

{

    AVCodecInternal *avci = avctx->internal;

    int planar, channels;

    int ret = 0;



    *got_frame_ptr = 0;



    avctx->pkt = avpkt;



    if (!avpkt->data && avpkt->size) {

        av_log(avctx, AV_LOG_ERROR, "invalid packet: NULL data, size != 0\n");

        return AVERROR(EINVAL);

    }



    apply_param_change(avctx, avpkt);



    avcodec_get_frame_defaults(frame);



    if (!avctx->refcounted_frames)

        av_frame_unref(&avci->to_free);



    if ((avctx->codec->capabilities & CODEC_CAP_DELAY) || avpkt->size) {

        ret = avctx->codec->decode(avctx, frame, got_frame_ptr, avpkt);

        if (ret >= 0 && *got_frame_ptr) {

            avctx->frame_number++;

            frame->pkt_dts = avpkt->dts;

            if (frame->format == AV_SAMPLE_FMT_NONE)

                frame->format = avctx->sample_fmt;



            if (!avctx->refcounted_frames) {

                avci->to_free = *frame;

                avci->to_free.extended_data = avci->to_free.data;

                memset(frame->buf, 0, sizeof(frame->buf));

                frame->extended_buf    = NULL;

                frame->nb_extended_buf = 0;

            }

        }



        if (ret < 0 && frame->data[0])

            av_frame_unref(frame);

    }



    /* many decoders assign whole AVFrames, thus overwriting extended_data;

     * make sure it's set correctly; assume decoders that actually use

     * extended_data are doing it correctly */

    planar   = av_sample_fmt_is_planar(frame->format);

    channels = av_get_channel_layout_nb_channels(frame->channel_layout);

    if (!(planar && channels > AV_NUM_DATA_POINTERS))

        frame->extended_data = frame->data;



    return ret;

}
