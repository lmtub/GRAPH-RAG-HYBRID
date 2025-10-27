static int transcode_video(InputStream *ist, AVPacket *pkt, int *got_output, int64_t *pkt_pts)

{

    AVFrame *decoded_frame, *filtered_frame = NULL;

    void *buffer_to_free = NULL;

    int i, ret = 0;

    float quality;

#if CONFIG_AVFILTER

    int frame_available = 1;

#endif



    if (!ist->decoded_frame && !(ist->decoded_frame = avcodec_alloc_frame()))

        return AVERROR(ENOMEM);

    else

        avcodec_get_frame_defaults(ist->decoded_frame);

    decoded_frame = ist->decoded_frame;

    pkt->pts  = *pkt_pts;

    pkt->dts  = ist->last_dts;

    *pkt_pts  = AV_NOPTS_VALUE;



    ret = avcodec_decode_video2(ist->st->codec,

                                decoded_frame, got_output, pkt);

    if (ret < 0)

        return ret;



    quality = same_quant ? decoded_frame->quality : 0;

    if (!*got_output) {

        /* no picture yet */

        return ret;

    }

    decoded_frame->pts = guess_correct_pts(&ist->pts_ctx, decoded_frame->pkt_pts,

                                           decoded_frame->pkt_dts);

    pkt->size = 0;

    pre_process_video_frame(ist, (AVPicture *)decoded_frame, &buffer_to_free);



    rate_emu_sleep(ist);



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = &output_streams[i];

        int frame_size, resample_changed;



        if (!check_output_constraints(ist, ost) || !ost->encoding_needed)

            continue;



#if CONFIG_AVFILTER

        resample_changed = ost->resample_width   != decoded_frame->width  ||

                           ost->resample_height  != decoded_frame->height ||

                           ost->resample_pix_fmt != decoded_frame->format;

        if (resample_changed) {

            av_log(NULL, AV_LOG_INFO,

                    "Input stream #%d:%d frame changed from size:%dx%d fmt:%s to size:%dx%d fmt:%s\n",

                    ist->file_index, ist->st->index,

                    ost->resample_width,  ost->resample_height,  av_get_pix_fmt_name(ost->resample_pix_fmt),

                    decoded_frame->width, decoded_frame->height, av_get_pix_fmt_name(decoded_frame->format));



            avfilter_graph_free(&ost->graph);

            if (configure_video_filters(ist, ost)) {

                av_log(NULL, AV_LOG_FATAL, "Error reinitializing filters!\n");

                exit_program(1);

            }



            ost->resample_width   = decoded_frame->width;

            ost->resample_height  = decoded_frame->height;

            ost->resample_pix_fmt = decoded_frame->format;

        }



        if (ist->st->sample_aspect_ratio.num)

            decoded_frame->sample_aspect_ratio = ist->st->sample_aspect_ratio;

        if (ist->st->codec->codec->capabilities & CODEC_CAP_DR1) {

            FrameBuffer      *buf = decoded_frame->opaque;

            AVFilterBufferRef *fb = avfilter_get_video_buffer_ref_from_arrays(

                                        decoded_frame->data, decoded_frame->linesize,

                                        AV_PERM_READ | AV_PERM_PRESERVE,

                                        ist->st->codec->width, ist->st->codec->height,

                                        ist->st->codec->pix_fmt);



            avfilter_copy_frame_props(fb, decoded_frame);

            fb->buf->priv           = buf;

            fb->buf->free           = filter_release_buffer;



            buf->refcount++;

            av_buffersrc_buffer(ost->input_video_filter, fb);

        } else

            av_vsrc_buffer_add_frame(ost->input_video_filter, decoded_frame,

                                     decoded_frame->pts, decoded_frame->sample_aspect_ratio);



        if (!ist->filtered_frame && !(ist->filtered_frame = avcodec_alloc_frame())) {

            av_free(buffer_to_free);

            return AVERROR(ENOMEM);

        } else

            avcodec_get_frame_defaults(ist->filtered_frame);

        filtered_frame = ist->filtered_frame;



        frame_available = avfilter_poll_frame(ost->output_video_filter->inputs[0]);

        while (frame_available) {

            AVRational ist_pts_tb;

            get_filtered_video_frame(ost->output_video_filter, filtered_frame, &ost->picref, &ist_pts_tb);

            if (ost->picref)

                filtered_frame->pts = av_rescale_q(ost->picref->pts, ist_pts_tb, AV_TIME_BASE_Q);

            if (ost->picref->video && !ost->frame_aspect_ratio)

                ost->st->codec->sample_aspect_ratio = ost->picref->video->pixel_aspect;

#else

            filtered_frame = decoded_frame;

#endif



            do_video_out(output_files[ost->file_index].ctx, ost, ist, filtered_frame, &frame_size,

                         same_quant ? quality : ost->st->codec->global_quality);

            if (vstats_filename && frame_size)

                do_video_stats(output_files[ost->file_index].ctx, ost, frame_size);

#if CONFIG_AVFILTER

            frame_available = ost->output_video_filter && avfilter_poll_frame(ost->output_video_filter->inputs[0]);

            if (ost->picref)

                avfilter_unref_buffer(ost->picref);

        }

#endif

    }



    av_free(buffer_to_free);

    return ret;

}
