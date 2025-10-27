static void do_video_out(AVFormatContext *s,
                         OutputStream *ost,
                         AVFrame *next_picture,
                         double sync_ipts)
{
    int ret, format_video_sync;
    AVPacket pkt;
    AVCodecContext *enc = ost->enc_ctx;
    AVCodecContext *mux_enc = ost->st->codec;
    int nb_frames, nb0_frames, i;
    double delta, delta0;
    double duration = 0;
    int frame_size = 0;
    InputStream *ist = NULL;
    AVFilterContext *filter = ost->filter->filter;
    if (ost->source_index >= 0)
        ist = input_streams[ost->source_index];
    if (filter->inputs[0]->frame_rate.num > 0 &&
        filter->inputs[0]->frame_rate.den > 0)
        duration = 1/(av_q2d(filter->inputs[0]->frame_rate) * av_q2d(enc->time_base));
    if(ist && ist->st->start_time != AV_NOPTS_VALUE && ist->st->first_dts != AV_NOPTS_VALUE && ost->frame_rate.num)
        duration = FFMIN(duration, 1/(av_q2d(ost->frame_rate) * av_q2d(enc->time_base)));
    if (!ost->filters_script &&
        !ost->filters &&
        next_picture &&
        ist &&
        lrintf(av_frame_get_pkt_duration(next_picture) * av_q2d(ist->st->time_base) / av_q2d(enc->time_base)) > 0) {
        duration = lrintf(av_frame_get_pkt_duration(next_picture) * av_q2d(ist->st->time_base) / av_q2d(enc->time_base));
    }
    if (!next_picture) {
        //end, flushing
        nb0_frames = nb_frames = mid_pred(ost->last_nb0_frames[0],
                                          ost->last_nb0_frames[1],
                                          ost->last_nb0_frames[2]);
    } else {
        delta0 = sync_ipts - ost->sync_opts;
        delta  = delta0 + duration;
        /* by default, we output a single frame */
        nb0_frames = 0;
        nb_frames = 1;
        format_video_sync = video_sync_method;
        if (format_video_sync == VSYNC_AUTO) {
            if(!strcmp(s->oformat->name, "avi")) {
                format_video_sync = VSYNC_VFR;
            } else
                format_video_sync = (s->oformat->flags & AVFMT_VARIABLE_FPS) ? ((s->oformat->flags & AVFMT_NOTIMESTAMPS) ? VSYNC_PASSTHROUGH : VSYNC_VFR) : VSYNC_CFR;
            if (   ist
                && format_video_sync == VSYNC_CFR
                && input_files[ist->file_index]->ctx->nb_streams == 1
                && input_files[ist->file_index]->input_ts_offset == 0) {
                format_video_sync = VSYNC_VSCFR;
            }
            if (format_video_sync == VSYNC_CFR && copy_ts) {
                format_video_sync = VSYNC_VSCFR;
            }
        }
        if (delta0 < 0 &&
            delta > 0 &&
            format_video_sync != VSYNC_PASSTHROUGH &&
            format_video_sync != VSYNC_DROP) {
            double cor = FFMIN(-delta0, duration);
            if (delta0 < -0.6) {
                av_log(NULL, AV_LOG_WARNING, "Past duration %f too large\n", -delta0);
            } else
                av_log(NULL, AV_LOG_DEBUG, "Cliping frame in rate conversion by %f\n", -delta0);
            sync_ipts += cor;
            duration -= cor;
            delta0 += cor;
        }
        switch (format_video_sync) {
        case VSYNC_VSCFR:
            if (ost->frame_number == 0 && delta - duration >= 0.5) {
                av_log(NULL, AV_LOG_DEBUG, "Not duplicating %d initial frames\n", (int)lrintf(delta - duration));
                delta = duration;
                delta0 = 0;
                ost->sync_opts = lrint(sync_ipts);
            }
        case VSYNC_CFR:
            // FIXME set to 0.5 after we fix some dts/pts bugs like in avidec.c
            if (frame_drop_threshold && delta < frame_drop_threshold && ost->frame_number) {
                nb_frames = 0;
            } else if (delta < -1.1)
                nb_frames = 0;
            else if (delta > 1.1) {
                nb_frames = lrintf(delta);
                if (delta0 > 1.1)
                    nb0_frames = lrintf(delta0 - 0.6);
            }
            break;
        case VSYNC_VFR:
            if (delta <= -0.6)
                nb_frames = 0;
            else if (delta > 0.6)
                ost->sync_opts = lrint(sync_ipts);
            break;
        case VSYNC_DROP:
        case VSYNC_PASSTHROUGH:
            ost->sync_opts = lrint(sync_ipts);
            break;
        default:
            av_assert0(0);
        }
    }
    nb_frames = FFMIN(nb_frames, ost->max_frames - ost->frame_number);
    nb0_frames = FFMIN(nb0_frames, nb_frames);
    memmove(ost->last_nb0_frames + 1,
            ost->last_nb0_frames,
            sizeof(ost->last_nb0_frames[0]) * (FF_ARRAY_ELEMS(ost->last_nb0_frames) - 1));
    ost->last_nb0_frames[0] = nb0_frames;
    if (nb0_frames == 0 && ost->last_droped) {
        nb_frames_drop++;
        av_log(NULL, AV_LOG_VERBOSE,
               "*** dropping frame %d from stream %d at ts %"PRId64"\n",
               ost->frame_number, ost->st->index, ost->last_frame->pts);
    }
    if (nb_frames > (nb0_frames && ost->last_droped) + (nb_frames > nb0_frames)) {
        if (nb_frames > dts_error_threshold * 30) {
            av_log(NULL, AV_LOG_ERROR, "%d frame duplication too large, skipping\n", nb_frames - 1);
            nb_frames_drop++;
        }
        nb_frames_dup += nb_frames - (nb0_frames && ost->last_droped) - (nb_frames > nb0_frames);
        av_log(NULL, AV_LOG_VERBOSE, "*** %d dup!\n", nb_frames - 1);
    }
    ost->last_droped = nb_frames == nb0_frames && next_picture;
  /* duplicates frame if needed */
  for (i = 0; i < nb_frames; i++) {
    AVFrame *in_picture;
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;
    if (i < nb0_frames && ost->last_frame) {
        in_picture = ost->last_frame;
    } else
        in_picture = next_picture;
    in_picture->pts = ost->sync_opts;
#if 1
    if (!check_recording_time(ost))
#else
    if (ost->frame_number >= ost->max_frames)
#endif
    if (s->oformat->flags & AVFMT_RAWPICTURE &&
        enc->codec->id == AV_CODEC_ID_RAWVIDEO) {
        /* raw pictures are written as AVPicture structure to
           avoid any copies. We support temporarily the older
           method. */
        if (in_picture->interlaced_frame)
            mux_enc->field_order = in_picture->top_field_first ? AV_FIELD_TB:AV_FIELD_BT;
        else
            mux_enc->field_order = AV_FIELD_PROGRESSIVE;
        pkt.data   = (uint8_t *)in_picture;
        pkt.size   =  sizeof(AVPicture);
        pkt.pts    = av_rescale_q(in_picture->pts, enc->time_base, ost->st->time_base);
        pkt.flags |= AV_PKT_FLAG_KEY;
        write_frame(s, &pkt, ost);
    } else {
        int got_packet, forced_keyframe = 0;
        double pts_time;
        if (enc->flags & (CODEC_FLAG_INTERLACED_DCT|CODEC_FLAG_INTERLACED_ME) &&
            ost->top_field_first >= 0)
            in_picture->top_field_first = !!ost->top_field_first;
        if (in_picture->interlaced_frame) {
            if (enc->codec->id == AV_CODEC_ID_MJPEG)
                mux_enc->field_order = in_picture->top_field_first ? AV_FIELD_TT:AV_FIELD_BB;
            else
                mux_enc->field_order = in_picture->top_field_first ? AV_FIELD_TB:AV_FIELD_BT;
        } else
            mux_enc->field_order = AV_FIELD_PROGRESSIVE;
        in_picture->quality = enc->global_quality;
        in_picture->pict_type = 0;
        pts_time = in_picture->pts != AV_NOPTS_VALUE ?
            in_picture->pts * av_q2d(enc->time_base) : NAN;
        if (ost->forced_kf_index < ost->forced_kf_count &&
            in_picture->pts >= ost->forced_kf_pts[ost->forced_kf_index]) {
            ost->forced_kf_index++;
            forced_keyframe = 1;
        } else if (ost->forced_keyframes_pexpr) {
            double res;
            ost->forced_keyframes_expr_const_values[FKF_T] = pts_time;
            res = av_expr_eval(ost->forced_keyframes_pexpr,
                               ost->forced_keyframes_expr_const_values, NULL);
            av_dlog(NULL, "force_key_frame: n:%f n_forced:%f prev_forced_n:%f t:%f prev_forced_t:%f -> res:%f\n",
                    ost->forced_keyframes_expr_const_values[FKF_N],
                    ost->forced_keyframes_expr_const_values[FKF_N_FORCED],
                    ost->forced_keyframes_expr_const_values[FKF_PREV_FORCED_N],
                    ost->forced_keyframes_expr_const_values[FKF_T],
                    ost->forced_keyframes_expr_const_values[FKF_PREV_FORCED_T],
                    res);
            if (res) {
                forced_keyframe = 1;
                ost->forced_keyframes_expr_const_values[FKF_PREV_FORCED_N] =
                    ost->forced_keyframes_expr_const_values[FKF_N];
                ost->forced_keyframes_expr_const_values[FKF_PREV_FORCED_T] =
                    ost->forced_keyframes_expr_const_values[FKF_T];
                ost->forced_keyframes_expr_const_values[FKF_N_FORCED] += 1;
            }
            ost->forced_keyframes_expr_const_values[FKF_N] += 1;
        }
        if (forced_keyframe) {
            in_picture->pict_type = AV_PICTURE_TYPE_I;
            av_log(NULL, AV_LOG_DEBUG, "Forced keyframe at time %f\n", pts_time);
        }
        update_benchmark(NULL);
        if (debug_ts) {
            av_log(NULL, AV_LOG_INFO, "encoder <- type:video "
                   "frame_pts:%s frame_pts_time:%s time_base:%d/%d\n",
                   av_ts2str(in_picture->pts), av_ts2timestr(in_picture->pts, &enc->time_base),
                   enc->time_base.num, enc->time_base.den);
        }
        ost->frames_encoded++;
        ret = avcodec_encode_video2(enc, &pkt, in_picture, &got_packet);
        update_benchmark("encode_video %d.%d", ost->file_index, ost->index);
        if (ret < 0) {
            av_log(NULL, AV_LOG_FATAL, "Video encoding failed\n");
            exit_program(1);
        }
        if (got_packet) {
            if (debug_ts) {
                av_log(NULL, AV_LOG_INFO, "encoder -> type:video "
                       "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s\n",
                       av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &enc->time_base),
                       av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &enc->time_base));
            }
            if (pkt.pts == AV_NOPTS_VALUE && !(enc->codec->capabilities & CODEC_CAP_DELAY))
                pkt.pts = ost->sync_opts;
            av_packet_rescale_ts(&pkt, enc->time_base, ost->st->time_base);
            if (debug_ts) {
                av_log(NULL, AV_LOG_INFO, "encoder -> type:video "
                    "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s\n",
                    av_ts2str(pkt.pts), av_ts2timestr(pkt.pts, &ost->st->time_base),
                    av_ts2str(pkt.dts), av_ts2timestr(pkt.dts, &ost->st->time_base));
            }
            frame_size = pkt.size;
            write_frame(s, &pkt, ost);
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
    if (vstats_filename && frame_size)
        do_video_stats(ost, frame_size);
  }
    if (!ost->last_frame)
        ost->last_frame = av_frame_alloc();
    av_frame_unref(ost->last_frame);
    if (next_picture)
        av_frame_ref(ost->last_frame, next_picture);
}