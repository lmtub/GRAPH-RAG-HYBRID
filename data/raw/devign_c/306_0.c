static int audio_decode_frame(VideoState *is, double *pts_ptr)

{

    AVPacket *pkt_temp = &is->audio_pkt_temp;

    AVPacket *pkt = &is->audio_pkt;

    AVCodecContext *dec = is->audio_st->codec;

    int len1, len2, data_size, resampled_data_size;

    int64_t dec_channel_layout;

    int got_frame;

    double pts;

    int new_packet = 0;

    int flush_complete = 0;

    int wanted_nb_samples;



    for (;;) {

        /* NOTE: the audio packet can contain several frames */

        while (pkt_temp->size > 0 || (!pkt_temp->data && new_packet)) {

            if (!is->frame) {

                if (!(is->frame = avcodec_alloc_frame()))

                    return AVERROR(ENOMEM);

            } else

                avcodec_get_frame_defaults(is->frame);



            if (is->paused)

                return -1;



            if (flush_complete)

                break;

            new_packet = 0;

            len1 = avcodec_decode_audio4(dec, is->frame, &got_frame, pkt_temp);

            if (len1 < 0) {

                /* if error, we skip the frame */

                pkt_temp->size = 0;

                break;

            }



            pkt_temp->data += len1;

            pkt_temp->size -= len1;



            if (!got_frame) {

                /* stop sending empty packets if the decoder is finished */

                if (!pkt_temp->data && dec->codec->capabilities & CODEC_CAP_DELAY)

                    flush_complete = 1;

                continue;

            }

            data_size = av_samples_get_buffer_size(NULL, dec->channels,

                                                   is->frame->nb_samples,

                                                   dec->sample_fmt, 1);



            dec_channel_layout = (dec->channel_layout && dec->channels == av_get_channel_layout_nb_channels(dec->channel_layout)) ? dec->channel_layout : av_get_default_channel_layout(dec->channels);

            wanted_nb_samples = synchronize_audio(is, is->frame->nb_samples);



            if (dec->sample_fmt != is->audio_src.fmt ||

                dec_channel_layout != is->audio_src.channel_layout ||

                dec->sample_rate != is->audio_src.freq ||

                (wanted_nb_samples != is->frame->nb_samples && !is->swr_ctx)) {

                if (is->swr_ctx)

                    swr_free(&is->swr_ctx);

                is->swr_ctx = swr_alloc_set_opts(NULL,

                                                 is->audio_tgt.channel_layout, is->audio_tgt.fmt, is->audio_tgt.freq,

                                                 dec_channel_layout,           dec->sample_fmt,   dec->sample_rate,

                                                 0, NULL);

                if (!is->swr_ctx || swr_init(is->swr_ctx) < 0) {

                    fprintf(stderr, "Cannot create sample rate converter for conversion of %d Hz %s %d channels to %d Hz %s %d channels!\n",

                        dec->sample_rate,

                        av_get_sample_fmt_name(dec->sample_fmt),

                        dec->channels,

                        is->audio_tgt.freq,

                        av_get_sample_fmt_name(is->audio_tgt.fmt),

                        is->audio_tgt.channels);

                    break;

                }

                is->audio_src.channel_layout = dec_channel_layout;

                is->audio_src.channels = dec->channels;

                is->audio_src.freq = dec->sample_rate;

                is->audio_src.fmt = dec->sample_fmt;

            }



            resampled_data_size = data_size;

            if (is->swr_ctx) {

                const uint8_t *in[] = { is->frame->data[0] };

                uint8_t *out[] = {is->audio_buf2};

                if (wanted_nb_samples != is->frame->nb_samples) {

                    if (swr_set_compensation(is->swr_ctx, (wanted_nb_samples - is->frame->nb_samples) * is->audio_tgt.freq / dec->sample_rate,

                                                wanted_nb_samples * is->audio_tgt.freq / dec->sample_rate) < 0) {

                        fprintf(stderr, "swr_set_compensation() failed\n");

                        break;

                    }

                }

                len2 = swr_convert(is->swr_ctx, out, sizeof(is->audio_buf2) / is->audio_tgt.channels / av_get_bytes_per_sample(is->audio_tgt.fmt),

                                                in, is->frame->nb_samples);

                if (len2 < 0) {

                    fprintf(stderr, "swr_convert() failed\n");

                    break;

                }

                if (len2 == sizeof(is->audio_buf2) / is->audio_tgt.channels / av_get_bytes_per_sample(is->audio_tgt.fmt)) {

                    fprintf(stderr, "warning: audio buffer is probably too small\n");

                    swr_init(is->swr_ctx);

                }

                is->audio_buf = is->audio_buf2;

                resampled_data_size = len2 * is->audio_tgt.channels * av_get_bytes_per_sample(is->audio_tgt.fmt);

            } else {

                is->audio_buf = is->frame->data[0];

            }



            /* if no pts, then compute it */

            pts = is->audio_clock;

            *pts_ptr = pts;

            is->audio_clock += (double)data_size /

                (dec->channels * dec->sample_rate * av_get_bytes_per_sample(dec->sample_fmt));

#ifdef DEBUG

            {

                static double last_clock;

                printf("audio: delay=%0.3f clock=%0.3f pts=%0.3f\n",

                       is->audio_clock - last_clock,

                       is->audio_clock, pts);

                last_clock = is->audio_clock;

            }

#endif

            return resampled_data_size;

        }



        /* free the current packet */

        if (pkt->data)

            av_free_packet(pkt);

        memset(pkt_temp, 0, sizeof(*pkt_temp));



        if (is->paused || is->audioq.abort_request) {

            return -1;

        }



        /* read next packet */

        if ((new_packet = packet_queue_get(&is->audioq, pkt, 1)) < 0)

            return -1;



        if (pkt->data == flush_pkt.data) {

            avcodec_flush_buffers(dec);

            flush_complete = 0;

        }



        *pkt_temp = *pkt;



        /* if update the audio clock with the pts */

        if (pkt->pts != AV_NOPTS_VALUE) {

            is->audio_clock = av_q2d(is->audio_st->time_base)*pkt->pts;

        }

    }

}
