static void do_video_out(AVFormatContext *s,

                         OutputStream *ost,

                         AVFrame *in_picture,

                         int *frame_size)

{

    int ret, format_video_sync;

    AVPacket pkt;

    AVCodecContext *enc = ost->enc_ctx;



    *frame_size = 0;



    format_video_sync = video_sync_method;

    if (format_video_sync == VSYNC_AUTO)

        format_video_sync = (s->oformat->flags & AVFMT_NOTIMESTAMPS) ? VSYNC_PASSTHROUGH :

                            (s->oformat->flags & AVFMT_VARIABLE_FPS) ? VSYNC_VFR : VSYNC_CFR;

    if (format_video_sync != VSYNC_PASSTHROUGH &&

        ost->frame_number &&

        in_picture->pts != AV_NOPTS_VALUE &&

        in_picture->pts < ost->sync_opts) {

        nb_frames_drop++;

        av_log(NULL, AV_LOG_WARNING,

               "*** dropping frame %d from stream %d at ts %"PRId64"\n",

               ost->frame_number, ost->st->index, in_picture->pts);

        return;

    }



    if (in_picture->pts == AV_NOPTS_VALUE)

        in_picture->pts = ost->sync_opts;

    ost->sync_opts = in_picture->pts;





    if (!ost->frame_number)

        ost->first_pts = in_picture->pts;



    av_init_packet(&pkt);

    pkt.data = NULL;

    pkt.size = 0;



    if (ost->frame_number >= ost->max_frames)

        return;



    if (enc->flags & (AV_CODEC_FLAG_INTERLACED_DCT | AV_CODEC_FLAG_INTERLACED_ME) &&

        ost->top_field_first >= 0)

        in_picture->top_field_first = !!ost->top_field_first;



    in_picture->quality = enc->global_quality;

    in_picture->pict_type = 0;

    if (ost->forced_kf_index < ost->forced_kf_count &&

        in_picture->pts >= ost->forced_kf_pts[ost->forced_kf_index]) {

        in_picture->pict_type = AV_PICTURE_TYPE_I;

        ost->forced_kf_index++;

    }



    ost->frames_encoded++;



    ret = avcodec_send_frame(enc, in_picture);

    if (ret < 0)

        goto error;



    /*

     * For video, there may be reordering, so we can't throw away frames on

     * encoder flush, we need to limit them here, before they go into encoder.

     */

    ost->frame_number++;



    while (1) {

        ret = avcodec_receive_packet(enc, &pkt);

        if (ret == AVERROR(EAGAIN))

            break;

        if (ret < 0)

            goto error;



        av_packet_rescale_ts(&pkt, enc->time_base, ost->st->time_base);

        output_packet(s, &pkt, ost);

        *frame_size = pkt.size;



        /* if two pass, output log */

        if (ost->logfile && enc->stats_out) {

            fprintf(ost->logfile, "%s", enc->stats_out);

        }



        ost->sync_opts++;

    }



    return;

error:

    av_assert0(ret != AVERROR(EAGAIN) && ret != AVERROR_EOF);

    av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");

    exit_program(1);

}
