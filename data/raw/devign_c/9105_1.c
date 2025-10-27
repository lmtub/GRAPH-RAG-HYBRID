static void do_video_out(AVFormatContext *s, OutputStream *ost,

                         InputStream *ist, AVFrame *in_picture)

{

    int nb_frames, i, ret, format_video_sync;

    AVFrame *final_picture;

    AVCodecContext *enc;

    double sync_ipts;

    double duration = 0;

    int frame_size = 0;

    float quality = same_quant ? in_picture->quality

                               : ost->st->codec->global_quality;



    enc = ost->st->codec;



    if (ist->st->start_time != AV_NOPTS_VALUE && ist->st->first_dts != AV_NOPTS_VALUE) {

        duration = FFMAX(av_q2d(ist->st->time_base), av_q2d(ist->st->codec->time_base));

        if(ist->st->avg_frame_rate.num)

            duration= FFMAX(duration, 1/av_q2d(ist->st->avg_frame_rate));



        duration /= av_q2d(enc->time_base);

    }



    sync_ipts = get_sync_ipts(ost, in_picture->pts) / av_q2d(enc->time_base);



    /* by default, we output a single frame */

    nb_frames = 1;



    format_video_sync = video_sync_method;

    if (format_video_sync == VSYNC_AUTO)

        format_video_sync = (s->oformat->flags & AVFMT_VARIABLE_FPS) ? ((s->oformat->flags & AVFMT_NOTIMESTAMPS) ? VSYNC_PASSTHROUGH : VSYNC_VFR) : 1;



    if (format_video_sync != VSYNC_PASSTHROUGH && format_video_sync != VSYNC_DROP) {

        double vdelta = sync_ipts - ost->sync_opts + duration;

        // FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c

        if (vdelta < -1.1)

            nb_frames = 0;

        else if (format_video_sync == VSYNC_VFR) {

            if (vdelta <= -0.6) {

                nb_frames = 0;

            } else if (vdelta > 0.6)

                ost->sync_opts = lrintf(sync_ipts);

        } else if (vdelta > 1.1)

            nb_frames = lrintf(vdelta);

        if (nb_frames == 0) {

            ++nb_frames_drop;

            av_log(NULL, AV_LOG_VERBOSE, "*** drop!\n");

        } else if (nb_frames > 1) {

            nb_frames_dup += nb_frames - 1;

            av_log(NULL, AV_LOG_VERBOSE, "*** %d dup!\n", nb_frames - 1);

        }

    } else

        ost->sync_opts = lrintf(sync_ipts);



    nb_frames = FFMIN(nb_frames, ost->max_frames - ost->frame_number);

    if (nb_frames <= 0)

        return;



    do_video_resample(ost, ist, in_picture, &final_picture);



    /* duplicates frame if needed */

    for (i = 0; i < nb_frames; i++) {

        AVPacket pkt;

        av_init_packet(&pkt);

        pkt.data = NULL;

        pkt.size = 0;



        if (s->oformat->flags & AVFMT_RAWPICTURE &&

            enc->codec->id == CODEC_ID_RAWVIDEO) {

            /* raw pictures are written as AVPicture structure to

               avoid any copies. We support temporarily the older

               method. */

            enc->coded_frame->interlaced_frame = in_picture->interlaced_frame;

            enc->coded_frame->top_field_first  = in_picture->top_field_first;

            pkt.data   = (uint8_t *)final_picture;

            pkt.size   =  sizeof(AVPicture);

            pkt.pts    = av_rescale_q(ost->sync_opts, enc->time_base, ost->st->time_base);

            pkt.flags |= AV_PKT_FLAG_KEY;



            write_frame(s, &pkt, ost);

        } else {

            int got_packet;

            AVFrame big_picture;



            big_picture = *final_picture;

            /* better than nothing: use input picture interlaced

               settings */

            big_picture.interlaced_frame = in_picture->interlaced_frame;

            if (ost->st->codec->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME)) {

                if (ost->top_field_first == -1)

                    big_picture.top_field_first = in_picture->top_field_first;

                else

                    big_picture.top_field_first = !!ost->top_field_first;

            }



            /* handles same_quant here. This is not correct because it may

               not be a global option */

            big_picture.quality = quality;

            if (!enc->me_threshold)

                big_picture.pict_type = 0;

            big_picture.pts = ost->sync_opts;

            if (ost->forced_kf_index < ost->forced_kf_count &&

                big_picture.pts >= ost->forced_kf_pts[ost->forced_kf_index]) {

                big_picture.pict_type = AV_PICTURE_TYPE_I;

                ost->forced_kf_index++;

            }

            ret = avcodec_encode_video2(enc, &pkt, &big_picture, &got_packet);

            if (ret < 0) {

                av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");

                exit_program(1);

            }



            if (got_packet) {

                if (pkt.pts == AV_NOPTS_VALUE && !(enc->codec->capabilities & CODEC_CAP_DELAY))

                    pkt.pts = ost->sync_opts;



                if (pkt.pts != AV_NOPTS_VALUE)

                    pkt.pts = av_rescale_q(pkt.pts, enc->time_base, ost->st->time_base);

                if (pkt.dts != AV_NOPTS_VALUE)

                    pkt.dts = av_rescale_q(pkt.dts, enc->time_base, ost->st->time_base);



                if (debug_ts) {

                    av_log(NULL, AV_LOG_INFO, "encoder -> type:video "

                           "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s\n",

                           av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &ost->st->time_base),

                           av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &ost->st->time_base));

                }



                if (format_video_sync == VSYNC_DROP)

                    pkt.pts = pkt.dts = AV_NOPTS_VALUE;



                write_frame(s, &pkt, ost);

                frame_size = pkt.size;

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

    if (vstats_filename && frame_size)

        do_video_stats(output_files[ost->file_index].ctx, ost, frame_size);

}