static int output_packet(InputStream *ist, int ist_index,

                         OutputStream *ost_table, int nb_ostreams,

                         const AVPacket *pkt)

{

    AVFormatContext *os;

    OutputStream *ost;

    int ret, i;

    int got_output;

    void *buffer_to_free = NULL;

    static unsigned int samples_size= 0;

    AVSubtitle subtitle, *subtitle_to_free;

    int64_t pkt_pts = AV_NOPTS_VALUE;

#if CONFIG_AVFILTER

    int frame_available;

#endif

    float quality;



    AVPacket avpkt;

    int bps = av_get_bytes_per_sample(ist->st->codec->sample_fmt);



    if(ist->next_pts == AV_NOPTS_VALUE)

        ist->next_pts= ist->pts;



    if (pkt == NULL) {

        /* EOF handling */

        av_init_packet(&avpkt);

        avpkt.data = NULL;

        avpkt.size = 0;

        goto handle_eof;

    } else {

        avpkt = *pkt;

    }



    if(pkt->dts != AV_NOPTS_VALUE)

        ist->next_pts = ist->pts = av_rescale_q(pkt->dts, ist->st->time_base, AV_TIME_BASE_Q);

    if(pkt->pts != AV_NOPTS_VALUE)

        pkt_pts = av_rescale_q(pkt->pts, ist->st->time_base, AV_TIME_BASE_Q);



    //while we have more to decode or while the decoder did output something on EOF

    while (avpkt.size > 0 || (!pkt && got_output)) {

        uint8_t *data_buf, *decoded_data_buf;

        int data_size, decoded_data_size;

        AVFrame *decoded_frame, *filtered_frame;

    handle_eof:

        ist->pts= ist->next_pts;



        if(avpkt.size && avpkt.size != pkt->size)

            av_log(NULL, ist->showed_multi_packet_warning ? AV_LOG_VERBOSE : AV_LOG_WARNING,

                   "Multiple frames in a packet from stream %d\n", pkt->stream_index);

            ist->showed_multi_packet_warning=1;



        /* decode the packet if needed */

        decoded_frame    = filtered_frame = NULL;

        decoded_data_buf = NULL; /* fail safe */

        decoded_data_size= 0;

        data_buf  = avpkt.data;

        data_size = avpkt.size;

        subtitle_to_free = NULL;

        if (ist->decoding_needed) {

            switch(ist->st->codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:{

                if(pkt && samples_size < FFMAX(pkt->size*sizeof(*samples), AVCODEC_MAX_AUDIO_FRAME_SIZE)) {

                    samples_size = FFMAX(pkt->size*sizeof(*samples), AVCODEC_MAX_AUDIO_FRAME_SIZE);

                    av_free(samples);

                    samples= av_malloc(samples_size);

                }

                decoded_data_size= samples_size;

                    /* XXX: could avoid copy if PCM 16 bits with same

                       endianness as CPU */

                ret = avcodec_decode_audio3(ist->st->codec, samples, &decoded_data_size,

                                            &avpkt);

                if (ret < 0)

                    return ret;

                avpkt.data += ret;

                avpkt.size -= ret;

                data_size   = ret;

                got_output  = decoded_data_size > 0;

                /* Some bug in mpeg audio decoder gives */

                /* decoded_data_size < 0, it seems they are overflows */

                if (!got_output) {

                    /* no audio frame */

                    continue;

                }

                decoded_data_buf = (uint8_t *)samples;

                ist->next_pts += ((int64_t)AV_TIME_BASE/bps * decoded_data_size) /

                    (ist->st->codec->sample_rate * ist->st->codec->channels);

                break;}

            case AVMEDIA_TYPE_VIDEO:

                    decoded_data_size = (ist->st->codec->width * ist->st->codec->height * 3) / 2;

                    if (!(decoded_frame = avcodec_alloc_frame()))

                        return AVERROR(ENOMEM);

                    avpkt.pts = pkt_pts;

                    avpkt.dts = ist->pts;

                    pkt_pts = AV_NOPTS_VALUE;



                    ret = avcodec_decode_video2(ist->st->codec,

                                                decoded_frame, &got_output, &avpkt);

                    quality = same_quant ? decoded_frame->quality : 0;

                    if (ret < 0)

                        goto fail;

                    if (!got_output) {

                        /* no picture yet */

                        av_freep(&decoded_frame);

                        goto discard_packet;

                    }

                    ist->next_pts = ist->pts = decoded_frame->best_effort_timestamp;

                    if (ist->st->codec->time_base.num != 0) {

                        int ticks= ist->st->parser ? ist->st->parser->repeat_pict+1 : ist->st->codec->ticks_per_frame;

                        ist->next_pts += ((int64_t)AV_TIME_BASE *

                                          ist->st->codec->time_base.num * ticks) /

                            ist->st->codec->time_base.den;

                    }

                    avpkt.size = 0;

                    buffer_to_free = NULL;

                    pre_process_video_frame(ist, (AVPicture *)decoded_frame, &buffer_to_free);

                    break;

            case AVMEDIA_TYPE_SUBTITLE:

                ret = avcodec_decode_subtitle2(ist->st->codec,

                                               &subtitle, &got_output, &avpkt);

                if (ret < 0)

                    return ret;

                if (!got_output) {

                    goto discard_packet;

                }

                subtitle_to_free = &subtitle;

                avpkt.size = 0;

                break;

            default:

                return -1;

            }

        } else {

            switch(ist->st->codec->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                ist->next_pts += ((int64_t)AV_TIME_BASE * ist->st->codec->frame_size) /

                    ist->st->codec->sample_rate;

                break;

            case AVMEDIA_TYPE_VIDEO:

                if (ist->st->codec->time_base.num != 0) {

                    int ticks= ist->st->parser ? ist->st->parser->repeat_pict+1 : ist->st->codec->ticks_per_frame;

                    ist->next_pts += ((int64_t)AV_TIME_BASE *

                                      ist->st->codec->time_base.num * ticks) /

                        ist->st->codec->time_base.den;

                }

                break;

            }

            avpkt.size = 0;

        }



#if CONFIG_AVFILTER

        if(ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

        for(i=0;i<nb_ostreams;i++) {

            OutputFile *of = &output_files[ost_table[i].file_index];

            if (of->start_time == 0 || ist->pts >= of->start_time) {

                ost = &ost_table[i];

                if (ost->input_video_filter && ost->source_index == ist_index) {

                if (!decoded_frame->sample_aspect_ratio.num)

                    decoded_frame->sample_aspect_ratio = ist->st->sample_aspect_ratio;

                decoded_frame->pts = ist->pts;



                av_vsrc_buffer_add_frame(ost->input_video_filter, decoded_frame, AV_VSRC_BUF_FLAG_OVERWRITE);

                }

            }

        }

#endif



        // preprocess audio (volume)

        if (ist->st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (audio_volume != 256) {

                short *volp;

                volp = samples;

                for(i=0;i<(decoded_data_size / sizeof(short));i++) {

                    int v = ((*volp) * audio_volume + 128) >> 8;

                    *volp++ = av_clip_int16(v);

                }

            }

        }



        /* frame rate emulation */

        if (input_files[ist->file_index].rate_emu) {

            int64_t pts = av_rescale(ist->pts, 1000000, AV_TIME_BASE);

            int64_t now = av_gettime() - ist->start;

            if (pts > now)

                usleep(pts - now);

        }

        /* if output time reached then transcode raw format,

           encode packets and output them */

        for (i = 0; i < nb_ostreams; i++) {

            OutputFile *of = &output_files[ost_table[i].file_index];

            int frame_size;



            ost = &ost_table[i];

            if (ost->source_index != ist_index)

                continue;



            if (of->start_time && ist->pts < of->start_time)

                continue;



            if (of->recording_time != INT64_MAX &&

                av_compare_ts(ist->pts, AV_TIME_BASE_Q, of->recording_time + of->start_time,

                              (AVRational){1, 1000000}) >= 0) {

                ost->is_past_recording_time = 1;

                continue;

            }



#if CONFIG_AVFILTER

            frame_available = ist->st->codec->codec_type != AVMEDIA_TYPE_VIDEO ||

                !ost->output_video_filter || avfilter_poll_frame(ost->output_video_filter->inputs[0]);

            while (frame_available) {

                if (ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO && ost->output_video_filter) {

                    AVRational ist_pts_tb = ost->output_video_filter->inputs[0]->time_base;

                    if (av_buffersink_get_buffer_ref(ost->output_video_filter, &ost->picref, 0) < 0)

                        goto cont;

                    if (!filtered_frame && !(filtered_frame = avcodec_alloc_frame())) {

                        ret = AVERROR(ENOMEM);

                        goto fail;

                    }

                    *filtered_frame= *decoded_frame; //for me_threshold

                    if (ost->picref) {

                        avfilter_fill_frame_from_video_buffer_ref(filtered_frame, ost->picref);

                        ist->pts = av_rescale_q(ost->picref->pts, ist_pts_tb, AV_TIME_BASE_Q);

                    }

                }

#else

                filtered_frame = decoded_frame;

#endif

                os = output_files[ost->file_index].ctx;



                /* set the input output pts pairs */

                //ost->sync_ipts = (double)(ist->pts + input_files[ist->file_index].ts_offset - start_time)/ AV_TIME_BASE;



                if (ost->encoding_needed) {

                    av_assert0(ist->decoding_needed);

                    switch(ost->st->codec->codec_type) {

                    case AVMEDIA_TYPE_AUDIO:

                        do_audio_out(os, ost, ist, decoded_data_buf, decoded_data_size);

                        break;

                    case AVMEDIA_TYPE_VIDEO:

#if CONFIG_AVFILTER

                        if (ost->picref->video && !ost->frame_aspect_ratio)

                            ost->st->codec->sample_aspect_ratio = ost->picref->video->sample_aspect_ratio;

#endif

                        do_video_out(os, ost, ist, filtered_frame, &frame_size,

                                     same_quant ? quality : ost->st->codec->global_quality);

                        if (vstats_filename && frame_size)

                            do_video_stats(os, ost, frame_size);

                        break;

                    case AVMEDIA_TYPE_SUBTITLE:

                        do_subtitle_out(os, ost, ist, &subtitle,

                                        pkt->pts);

                        break;

                    default:

                        abort();

                    }

                } else {

                    AVPicture pict;

                    AVPacket opkt;

                    int64_t ost_tb_start_time= av_rescale_q(of->start_time, AV_TIME_BASE_Q, ost->st->time_base);

                    av_init_packet(&opkt);



                    if ((!ost->frame_number && !(pkt->flags & AV_PKT_FLAG_KEY)) && !copy_initial_nonkeyframes)

#if !CONFIG_AVFILTER

                        continue;

#else

                        goto cont;

#endif



                    /* no reencoding needed : output the packet directly */

                    /* force the input stream PTS */



                    if(ost->st->codec->codec_type == AVMEDIA_TYPE_AUDIO)

                        audio_size += data_size;

                    else if (ost->st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

                        video_size += data_size;

                        ost->sync_opts++;

                    }



                    opkt.stream_index= ost->index;

                    if(pkt->pts != AV_NOPTS_VALUE)

                        opkt.pts= av_rescale_q(pkt->pts, ist->st->time_base, ost->st->time_base) - ost_tb_start_time;

                    else

                        opkt.pts= AV_NOPTS_VALUE;



                    if (pkt->dts == AV_NOPTS_VALUE)

                        opkt.dts = av_rescale_q(ist->pts, AV_TIME_BASE_Q, ost->st->time_base);

                    else

                        opkt.dts = av_rescale_q(pkt->dts, ist->st->time_base, ost->st->time_base);

                    opkt.dts -= ost_tb_start_time;



                    opkt.duration = av_rescale_q(pkt->duration, ist->st->time_base, ost->st->time_base);

                    opkt.flags= pkt->flags;



                    //FIXME remove the following 2 lines they shall be replaced by the bitstream filters

                    if(   ost->st->codec->codec_id != CODEC_ID_H264

                       && ost->st->codec->codec_id != CODEC_ID_MPEG1VIDEO

                       && ost->st->codec->codec_id != CODEC_ID_MPEG2VIDEO

                       ) {

                        if(av_parser_change(ist->st->parser, ost->st->codec, &opkt.data, &opkt.size, data_buf, data_size, pkt->flags & AV_PKT_FLAG_KEY))

                            opkt.destruct= av_destruct_packet;

                    } else {

                        opkt.data = data_buf;

                        opkt.size = data_size;

                    }



                    if (os->oformat->flags & AVFMT_RAWPICTURE) {

                        /* store AVPicture in AVPacket, as expected by the output format */

                        avpicture_fill(&pict, opkt.data, ost->st->codec->pix_fmt, ost->st->codec->width, ost->st->codec->height);

                        opkt.data = (uint8_t *)&pict;

                        opkt.size = sizeof(AVPicture);

                        opkt.flags |= AV_PKT_FLAG_KEY;

                    }

                    write_frame(os, &opkt, ost->st->codec, ost->bitstream_filters);

                    ost->st->codec->frame_number++;

                    ost->frame_number++;

                    av_free_packet(&opkt);

                }

#if CONFIG_AVFILTER

                cont:

                frame_available = (ist->st->codec->codec_type == AVMEDIA_TYPE_VIDEO) &&

                                   ost->output_video_filter && avfilter_poll_frame(ost->output_video_filter->inputs[0]);

                avfilter_unref_buffer(ost->picref);

            }

            av_freep(&filtered_frame);

#endif

        }



fail:

        av_free(buffer_to_free);

        /* XXX: allocate the subtitles in the codec ? */

        if (subtitle_to_free) {

            avsubtitle_free(subtitle_to_free);

            subtitle_to_free = NULL;

        }

        av_freep(&decoded_frame);

        if (ret < 0)

            return ret;

    }

 discard_packet:



    return 0;

}
