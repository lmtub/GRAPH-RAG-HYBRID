static int reap_filters(int flush)

{

    AVFrame *filtered_frame = NULL;

    int i;



    /* Reap all buffers present in the buffer sinks */

    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];

        OutputFile    *of = output_files[ost->file_index];

        AVFilterContext *filter;

        AVCodecContext *enc = ost->enc_ctx;

        int ret = 0;



        if (!ost->filter || !ost->filter->graph->graph)

            continue;

        filter = ost->filter->filter;



        if (!ost->initialized) {

            char error[1024];

            ret = init_output_stream(ost, error, sizeof(error));

            if (ret < 0) {

                av_log(NULL, AV_LOG_ERROR, "Error initializing output stream %d:%d -- %s\n",

                       ost->file_index, ost->index, error);

                exit_program(1);

            }

        }



        if (!ost->filtered_frame && !(ost->filtered_frame = av_frame_alloc())) {

            return AVERROR(ENOMEM);

        }

        filtered_frame = ost->filtered_frame;



        while (1) {

            double float_pts = AV_NOPTS_VALUE; // this is identical to filtered_frame.pts but with higher precision

            ret = av_buffersink_get_frame_flags(filter, filtered_frame,

                                               AV_BUFFERSINK_FLAG_NO_REQUEST);

            if (ret < 0) {

                if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {

                    av_log(NULL, AV_LOG_WARNING,

                           "Error in av_buffersink_get_frame_flags(): %s\n", av_err2str(ret));

                } else if (flush && ret == AVERROR_EOF) {

                    if (av_buffersink_get_type(filter) == AVMEDIA_TYPE_VIDEO)

                        do_video_out(of, ost, NULL, AV_NOPTS_VALUE);

                }

                break;

            }

            if (ost->finished) {

                av_frame_unref(filtered_frame);

                continue;

            }

            if (filtered_frame->pts != AV_NOPTS_VALUE) {

                int64_t start_time = (of->start_time == AV_NOPTS_VALUE) ? 0 : of->start_time;

                AVRational filter_tb = av_buffersink_get_time_base(filter);

                AVRational tb = enc->time_base;

                int extra_bits = av_clip(29 - av_log2(tb.den), 0, 16);



                tb.den <<= extra_bits;

                float_pts =

                    av_rescale_q(filtered_frame->pts, filter_tb, tb) -

                    av_rescale_q(start_time, AV_TIME_BASE_Q, tb);

                float_pts /= 1 << extra_bits;

                // avoid exact midoints to reduce the chance of rounding differences, this can be removed in case the fps code is changed to work with integers

                float_pts += FFSIGN(float_pts) * 1.0 / (1<<17);



                filtered_frame->pts =

                    av_rescale_q(filtered_frame->pts, filter_tb, enc->time_base) -

                    av_rescale_q(start_time, AV_TIME_BASE_Q, enc->time_base);

            }

            //if (ost->source_index >= 0)

            //    *filtered_frame= *input_streams[ost->source_index]->decoded_frame; //for me_threshold



            switch (av_buffersink_get_type(filter)) {

            case AVMEDIA_TYPE_VIDEO:

                if (!ost->frame_aspect_ratio.num)

                    enc->sample_aspect_ratio = filtered_frame->sample_aspect_ratio;



                if (debug_ts) {

                    av_log(NULL, AV_LOG_INFO, "filter -> pts:%s pts_time:%s exact:%f time_base:%d/%d\n",

                            av_ts2str(filtered_frame->pts), av_ts2timestr(filtered_frame->pts, &enc->time_base),

                            float_pts,

                            enc->time_base.num, enc->time_base.den);

                }



                do_video_out(of, ost, filtered_frame, float_pts);

                break;

            case AVMEDIA_TYPE_AUDIO:

                if (!(enc->codec->capabilities & AV_CODEC_CAP_PARAM_CHANGE) &&

                    enc->channels != av_frame_get_channels(filtered_frame)) {

                    av_log(NULL, AV_LOG_ERROR,

                           "Audio filter graph output is not normalized and encoder does not support parameter changes\n");

                    break;

                }

                do_audio_out(of, ost, filtered_frame);

                break;

            default:

                // TODO support subtitle filters

                av_assert0(0);

            }



            av_frame_unref(filtered_frame);

        }

    }



    return 0;

}
