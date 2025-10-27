static int poll_filters(void)

{

    AVFilterBufferRef *picref;

    AVFrame *filtered_frame = NULL;

    int i, frame_size;



    for (i = 0; i < nb_output_streams; i++) {

        OutputStream *ost = output_streams[i];

        OutputFile    *of = output_files[ost->file_index];

        int ret = 0;



        if (!ost->filter)

            continue;



        if (!ost->filtered_frame && !(ost->filtered_frame = avcodec_alloc_frame())) {

            return AVERROR(ENOMEM);

        } else

            avcodec_get_frame_defaults(ost->filtered_frame);

        filtered_frame = ost->filtered_frame;



        while (ret >= 0 && !ost->is_past_recording_time) {

            if (ost->enc->type == AVMEDIA_TYPE_AUDIO &&

                !(ost->enc->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE))

                ret = av_buffersink_read_samples(ost->filter->filter, &picref,

                                                 ost->st->codec->frame_size);

            else

                ret = av_buffersink_read(ost->filter->filter, &picref);



            if (ret < 0)

                break;



            avfilter_copy_buf_props(filtered_frame, picref);

            if (picref->pts != AV_NOPTS_VALUE)

                filtered_frame->pts = av_rescale_q(picref->pts,

                                                   ost->filter->filter->inputs[0]->time_base,

                                                   ost->st->codec->time_base) -

                                      av_rescale_q(of->start_time,

                                                   AV_TIME_BASE_Q,

                                                   ost->st->codec->time_base);



            if (of->start_time && filtered_frame->pts < of->start_time) {

                avfilter_unref_buffer(picref);

                continue;

            }



            switch (ost->filter->filter->inputs[0]->type) {

            case AVMEDIA_TYPE_VIDEO:

                if (!ost->frame_aspect_ratio)

                    ost->st->codec->sample_aspect_ratio = picref->video->pixel_aspect;



                do_video_out(of->ctx, ost, filtered_frame, &frame_size,

                             same_quant ? ost->last_quality :

                                          ost->st->codec->global_quality);

                if (vstats_filename && frame_size)

                    do_video_stats(of->ctx, ost, frame_size);

                break;

            case AVMEDIA_TYPE_AUDIO:

                do_audio_out(of->ctx, ost, filtered_frame);

                break;

            default:

                // TODO support subtitle filters

                av_assert0(0);

            }



            avfilter_unref_buffer(picref);

        }

    }

    return 0;

}
