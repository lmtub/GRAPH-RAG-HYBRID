static void do_video_out(AVFormatContext *s,

                         OutputStream *ost,

                         AVFrame *in_picture,

                         int *frame_size)

{

    int ret, format_video_sync;

    AVPacket pkt;

    AVCodecContext *enc = ost->st->codec;



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

        av_log(NULL, AV_LOG_VERBOSE, "*** drop!\n");

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



    if (s->oformat->flags & AVFMT_RAWPICTURE &&

        enc->codec->id == AV_CODEC_ID_RAWVIDEO) {

        /* raw pictures are written as AVPicture structure to

           avoid any copies. We support temporarily the older

           method. */

        enc->coded_frame->interlaced_frame = in_picture->interlaced_frame;

        enc->coded_frame->top_field_first  = in_picture->top_field_first;

        pkt.data   = (uint8_t *)in_picture;

        pkt.size   =  sizeof(AVPicture);

        pkt.pts    = av_rescale_q(in_picture->pts, enc->time_base, ost->st->time_base);

        pkt.flags |= AV_PKT_FLAG_KEY;



        write_frame(s, &pkt, ost);

    } else {

        int got_packet;



        if (ost->st->codec->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME) &&

            ost->top_field_first >= 0)

            in_picture->top_field_first = !!ost->top_field_first;



        in_picture->quality = ost->st->codec->global_quality;

        if (!enc->me_threshold)

            in_picture->pict_type = 0;

        if (ost->forced_kf_index < ost->forced_kf_count &&

            in_picture->pts >= ost->forced_kf_pts[ost->forced_kf_index]) {

            in_picture->pict_type = AV_PICTURE_TYPE_I;

            ost->forced_kf_index++;

        }

        ret = avcodec_encode_video2(enc, &pkt, in_picture, &got_packet);

        if (ret < 0) {

            av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");

            exit(1);

        }



        if (got_packet) {

            if (pkt.pts != AV_NOPTS_VALUE)

                pkt.pts = av_rescale_q(pkt.pts, enc->time_base, ost->st->time_base);

            if (pkt.dts != AV_NOPTS_VALUE)

                pkt.dts = av_rescale_q(pkt.dts, enc->time_base, ost->st->time_base);



            write_frame(s, &pkt, ost);

            *frame_size = pkt.size;

            video_size += pkt.size;



            /* if two pass, output log */

            if (ost->logfile && enc->stats_out) {

                fprintf(ost->logfile, "%s", enc->stats_out);

            }

        }

    }

    ost->sync_opts++;

    /*

     * For video, number of frames in == number of packets out.

     * But there may be reordering, so we can't throw away frames on encoder

     * flush, we need to limit them here, before they go into encoder.

     */

    ost->frame_number++;

}
