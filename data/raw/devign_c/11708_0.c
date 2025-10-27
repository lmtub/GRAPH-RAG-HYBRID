static int decode_audio(InputStream *ist, AVPacket *pkt, int *got_output)

{

    AVFrame *decoded_frame, *f;

    AVCodecContext *avctx = ist->dec_ctx;

    int i, ret, err = 0, resample_changed;

    AVRational decoded_frame_tb;



    if (!ist->decoded_frame && !(ist->decoded_frame = av_frame_alloc()))

        return AVERROR(ENOMEM);

    if (!ist->filter_frame && !(ist->filter_frame = av_frame_alloc()))

        return AVERROR(ENOMEM);

    decoded_frame = ist->decoded_frame;



    update_benchmark(NULL);

    ret = avcodec_decode_audio4(avctx, decoded_frame, got_output, pkt);

    update_benchmark("decode_audio %d.%d", ist->file_index, ist->st->index);



    if (ret >= 0 && avctx->sample_rate <= 0) {

        av_log(avctx, AV_LOG_ERROR, "Sample rate %d invalid\n", avctx->sample_rate);

        ret = AVERROR_INVALIDDATA;

    }



    if (*got_output || ret<0)

        decode_error_stat[ret<0] ++;



    if (ret < 0 && exit_on_error)

        exit_program(1);



    if (!*got_output || ret < 0)

        return ret;



    ist->samples_decoded += decoded_frame->nb_samples;

    ist->frames_decoded++;



#if 1

    /* increment next_dts to use for the case where the input stream does not

       have timestamps or there are multiple frames in the packet */

    ist->next_pts += ((int64_t)AV_TIME_BASE * decoded_frame->nb_samples) /

                     avctx->sample_rate;

    ist->next_dts += ((int64_t)AV_TIME_BASE * decoded_frame->nb_samples) /

                     avctx->sample_rate;

#endif



    resample_changed = ist->resample_sample_fmt     != decoded_frame->format         ||

                       ist->resample_channels       != avctx->channels               ||

                       ist->resample_channel_layout != decoded_frame->channel_layout ||

                       ist->resample_sample_rate    != decoded_frame->sample_rate;

    if (resample_changed) {

        char layout1[64], layout2[64];



        if (!guess_input_channel_layout(ist)) {

            av_log(NULL, AV_LOG_FATAL, "Unable to find default channel "

                   "layout for Input Stream #%d.%d\n", ist->file_index,

                   ist->st->index);

            exit_program(1);

        }

        decoded_frame->channel_layout = avctx->channel_layout;



        av_get_channel_layout_string(layout1, sizeof(layout1), ist->resample_channels,

                                     ist->resample_channel_layout);

        av_get_channel_layout_string(layout2, sizeof(layout2), avctx->channels,

                                     decoded_frame->channel_layout);



        av_log(NULL, AV_LOG_INFO,

               "Input stream #%d:%d frame changed from rate:%d fmt:%s ch:%d chl:%s to rate:%d fmt:%s ch:%d chl:%s\n",

               ist->file_index, ist->st->index,

               ist->resample_sample_rate,  av_get_sample_fmt_name(ist->resample_sample_fmt),

               ist->resample_channels, layout1,

               decoded_frame->sample_rate, av_get_sample_fmt_name(decoded_frame->format),

               avctx->channels, layout2);



        ist->resample_sample_fmt     = decoded_frame->format;

        ist->resample_sample_rate    = decoded_frame->sample_rate;

        ist->resample_channel_layout = decoded_frame->channel_layout;

        ist->resample_channels       = avctx->channels;



        for (i = 0; i < nb_filtergraphs; i++)

            if (ist_in_filtergraph(filtergraphs[i], ist)) {

                FilterGraph *fg = filtergraphs[i];

                if (configure_filtergraph(fg) < 0) {

                    av_log(NULL, AV_LOG_FATAL, "Error reinitializing filters!\n");

                    exit_program(1);

                }

            }

    }



    /* if the decoder provides a pts, use it instead of the last packet pts.

       the decoder could be delaying output by a packet or more. */

    if (decoded_frame->pts != AV_NOPTS_VALUE) {

        ist->dts = ist->next_dts = ist->pts = ist->next_pts = av_rescale_q(decoded_frame->pts, avctx->time_base, AV_TIME_BASE_Q);

        decoded_frame_tb   = avctx->time_base;

    } else if (decoded_frame->pkt_pts != AV_NOPTS_VALUE) {

        decoded_frame->pts = decoded_frame->pkt_pts;

        decoded_frame_tb   = ist->st->time_base;

    } else if (pkt->pts != AV_NOPTS_VALUE) {

        decoded_frame->pts = pkt->pts;

        decoded_frame_tb   = ist->st->time_base;

    }else {

        decoded_frame->pts = ist->dts;

        decoded_frame_tb   = AV_TIME_BASE_Q;

    }

    pkt->pts           = AV_NOPTS_VALUE;

    if (decoded_frame->pts != AV_NOPTS_VALUE)

        decoded_frame->pts = av_rescale_delta(decoded_frame_tb, decoded_frame->pts,

                                              (AVRational){1, avctx->sample_rate}, decoded_frame->nb_samples, &ist->filter_in_rescale_delta_last,

                                              (AVRational){1, avctx->sample_rate});

    ist->nb_samples = decoded_frame->nb_samples;

    for (i = 0; i < ist->nb_filters; i++) {

        if (i < ist->nb_filters - 1) {

            f = ist->filter_frame;

            err = av_frame_ref(f, decoded_frame);

            if (err < 0)

                break;

        } else

            f = decoded_frame;

        err = av_buffersrc_add_frame_flags(ist->filters[i]->filter, f,

                                     AV_BUFFERSRC_FLAG_PUSH);

        if (err == AVERROR_EOF)

            err = 0; /* ignore */

        if (err < 0)

            break;

    }

    decoded_frame->pts = AV_NOPTS_VALUE;



    av_frame_unref(ist->filter_frame);

    av_frame_unref(decoded_frame);

    return err < 0 ? err : ret;

}
