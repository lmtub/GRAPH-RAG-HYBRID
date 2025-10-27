static int poll_filter(OutputStream *ost)

{

    OutputFile    *of = output_files[ost->file_index];

    AVFrame *filtered_frame = NULL;

    int frame_size, ret;



    if (!ost->filtered_frame && !(ost->filtered_frame = avcodec_alloc_frame())) {

        return AVERROR(ENOMEM);

    } else

        avcodec_get_frame_defaults(ost->filtered_frame);

    filtered_frame = ost->filtered_frame;



    if (ost->enc->type == AVMEDIA_TYPE_AUDIO &&

        !(ost->enc->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE))

        ret = av_buffersink_get_samples(ost->filter->filter, filtered_frame,

                                         ost->st->codec->frame_size);

    else

        ret = av_buffersink_get_frame(ost->filter->filter, filtered_frame);



    if (ret < 0)

        return ret;



    if (filtered_frame->pts != AV_NOPTS_VALUE) {

        filtered_frame->pts = av_rescale_q(filtered_frame->pts,

                                           ost->filter->filter->inputs[0]->time_base,

                                           ost->st->codec->time_base) -

                              av_rescale_q(of->start_time,

                                           AV_TIME_BASE_Q,

                                           ost->st->codec->time_base);

    }



    switch (ost->filter->filter->inputs[0]->type) {

    case AVMEDIA_TYPE_VIDEO:

        if (!ost->frame_aspect_ratio)

            ost->st->codec->sample_aspect_ratio = filtered_frame->sample_aspect_ratio;



        do_video_out(of->ctx, ost, filtered_frame, &frame_size);

        if (vstats_filename && frame_size)

            do_video_stats(ost, frame_size);

        break;

    case AVMEDIA_TYPE_AUDIO:

        do_audio_out(of->ctx, ost, filtered_frame);

        break;

    default:

        // TODO support subtitle filters

        av_assert0(0);

    }



    av_frame_unref(filtered_frame);



    return 0;

}
