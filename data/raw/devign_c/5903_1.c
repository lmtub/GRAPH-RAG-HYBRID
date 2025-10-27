static int transcode_video(InputStream *ist, AVPacket *pkt, int *got_output, int64_t *pkt_pts, int64_t *pkt_dts)

{

    AVFrame *decoded_frame, *filtered_frame = NULL;

    void *buffer_to_free = NULL;

    int i, ret = 0;

    float quality = 0;

#if CONFIG_AVFILTER

    int frame_available = 1;

#endif

    int duration=0;

    int64_t *best_effort_timestamp;

    AVRational *frame_sample_aspect;



    if (!ist->decoded_frame && !(ist->decoded_frame = avcodec_alloc_frame()))

        return AVERROR(ENOMEM);

    else

        avcodec_get_frame_defaults(ist->decoded_frame);

    decoded_frame = ist->decoded_frame;

    pkt->pts  = *pkt_pts;

    pkt->dts  = *pkt_dts;

    *pkt_pts  = AV_NOPTS_VALUE;



    if (pkt->duration) {

        duration = av_rescale_q(pkt->duration, ist->st->time_base, AV_TIME_BASE_Q);

    } else if(ist->st->codec->time_base.num != 0) {

        int ticks= ist->st->parser ? ist->st->parser->repeat_pict+1 : ist->st->codec->ticks_per_frame;

        duration = ((int64_t)AV_TIME_BASE *

                          ist->st->codec->time_base.num * ticks) /

                          ist->st->codec->time_base.den;

    }



    if(*pkt_dts != AV_NOPTS_VALUE && duration) {

        *pkt_dts += duration;

    }else

        *pkt_dts = AV_NOPTS_VALUE;



    ret = avcodec_decode_video2(ist->st->codec,

                                decoded_frame, got_output, pkt);

    if (ret < 0)

        return ret;



    quality = same_quant ? decoded_frame->quality : 0;

    if (!*got_output) {

        /* no picture yet */

        return ret;

    }



    best_effort_timestamp= av_opt_ptr(avcodec_get_frame_class(), decoded_frame, "best_effort_timestamp");

    if(*best_effort_timestamp != AV_NOPTS_VALUE)

        ist->next_pts = ist->pts = *best_effort_timestamp;



    ist->next_pts += duration;

    pkt->size = 0;



    pre_process_video_frame(ist, (AVPicture *)decoded_frame, &buffer_to_free);



#if CONFIG_AVFILTER

    frame_sample_aspect= av_opt_ptr(avcodec_get_frame_class(), decoded_frame, "sample_aspect_ratio");

    for(i=0;i<nb_output_streams;i++) {

        OutputStream *ost = ost = &output_streams[i];

        if(check_output_constraints(ist, ost)){

            if (!frame_sample_aspect->num)

                *frame_sample_aspect = ist->st->sample_aspect_ratio;

            decoded_frame->pts = ist->pts;



            av_vsrc_buffer_add_frame(ost->input_video_filter, decoded_frame, AV_VSRC_BUF_FLAG_OVERWRITE);

        }

    }

#endif



    rate_emu_sleep(ist);



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = &output_streams[i];

        int frame_size;



        if (!check_output_constraints(ist, ost) || !ost->encoding_needed)

            continue;



#if CONFIG_AVFILTER

        if (ost->input_video_filter) {

            frame_available = av_buffersink_poll_frame(ost->output_video_filter);

        }

        while (frame_available) {

            if (ost->output_video_filter) {

                AVRational ist_pts_tb = ost->output_video_filter->inputs[0]->time_base;

                if (av_buffersink_get_buffer_ref(ost->output_video_filter, &ost->picref, 0) < 0){

                    av_log(0, AV_LOG_WARNING, "AV Filter told us it has a frame available but failed to output one\n");

                    goto cont;

                }

                if (!ist->filtered_frame && !(ist->filtered_frame = avcodec_alloc_frame())) {

                    av_free(buffer_to_free);

                    return AVERROR(ENOMEM);

                } else

                    avcodec_get_frame_defaults(ist->filtered_frame);

                filtered_frame = ist->filtered_frame;

                *filtered_frame= *decoded_frame; //for me_threshold

                if (ost->picref) {

                    avfilter_fill_frame_from_video_buffer_ref(filtered_frame, ost->picref);

                    ist->pts = av_rescale_q(ost->picref->pts, ist_pts_tb, AV_TIME_BASE_Q);

                }

            }

            if (ost->picref->video && !ost->frame_aspect_ratio)

                ost->st->codec->sample_aspect_ratio = ost->picref->video->sample_aspect_ratio;

#else

            filtered_frame = decoded_frame;

#endif



            do_video_out(output_files[ost->file_index].ctx, ost, ist, filtered_frame, &frame_size,

                         same_quant ? quality : ost->st->codec->global_quality);

            if (vstats_filename && frame_size)

                do_video_stats(output_files[ost->file_index].ctx, ost, frame_size);

#if CONFIG_AVFILTER

            cont:

            frame_available = ost->output_video_filter && av_buffersink_poll_frame(ost->output_video_filter);

            avfilter_unref_buffer(ost->picref);

        }

#endif

    }



    av_free(buffer_to_free);

    return ret;

}
