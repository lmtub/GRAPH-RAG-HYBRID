static int decode_thread(void *arg)

{

    VideoState *is = arg;

    AVFormatContext *ic;

    int err, i, ret;

    int st_index[AVMEDIA_TYPE_NB];

    AVPacket pkt1, *pkt = &pkt1;

    AVFormatParameters params, *ap = &params;

    int eof=0;

    int pkt_in_play_range = 0;



    ic = avformat_alloc_context();



    memset(st_index, -1, sizeof(st_index));

    is->video_stream = -1;

    is->audio_stream = -1;

    is->subtitle_stream = -1;



    global_video_state = is;

    url_set_interrupt_cb(decode_interrupt_cb);



    memset(ap, 0, sizeof(*ap));



    ap->prealloced_context = 1;

    ap->width = frame_width;

    ap->height= frame_height;

    ap->time_base= (AVRational){1, 25};

    ap->pix_fmt = frame_pix_fmt;



    set_context_opts(ic, avformat_opts, AV_OPT_FLAG_DECODING_PARAM, NULL);



    err = av_open_input_file(&ic, is->filename, is->iformat, 0, ap);

    if (err < 0) {

        print_error(is->filename, err);

        ret = -1;

        goto fail;

    }

    is->ic = ic;



    if(genpts)

        ic->flags |= AVFMT_FLAG_GENPTS;



    err = av_find_stream_info(ic);

    if (err < 0) {

        fprintf(stderr, "%s: could not find codec parameters\n", is->filename);

        ret = -1;

        goto fail;

    }

    if(ic->pb)

        ic->pb->eof_reached= 0; //FIXME hack, ffplay maybe should not use url_feof() to test for the end



    if(seek_by_bytes<0)

        seek_by_bytes= !!(ic->iformat->flags & AVFMT_TS_DISCONT);



    /* if seeking requested, we execute it */

    if (start_time != AV_NOPTS_VALUE) {

        int64_t timestamp;



        timestamp = start_time;

        /* add the stream start time */

        if (ic->start_time != AV_NOPTS_VALUE)

            timestamp += ic->start_time;

        ret = avformat_seek_file(ic, -1, INT64_MIN, timestamp, INT64_MAX, 0);

        if (ret < 0) {

            fprintf(stderr, "%s: could not seek to position %0.3f\n",

                    is->filename, (double)timestamp / AV_TIME_BASE);

        }

    }



    for (i = 0; i < ic->nb_streams; i++)

        ic->streams[i]->discard = AVDISCARD_ALL;

    if (!video_disable)

        st_index[AVMEDIA_TYPE_VIDEO] =

            av_find_best_stream(ic, AVMEDIA_TYPE_VIDEO,

                                wanted_stream[AVMEDIA_TYPE_VIDEO], -1, NULL, 0);

    if (!audio_disable)

        st_index[AVMEDIA_TYPE_AUDIO] =

            av_find_best_stream(ic, AVMEDIA_TYPE_AUDIO,

                                wanted_stream[AVMEDIA_TYPE_AUDIO],

                                st_index[AVMEDIA_TYPE_VIDEO],

                                NULL, 0);

    if (!video_disable)

        st_index[AVMEDIA_TYPE_SUBTITLE] =

            av_find_best_stream(ic, AVMEDIA_TYPE_SUBTITLE,

                                wanted_stream[AVMEDIA_TYPE_SUBTITLE],

                                (st_index[AVMEDIA_TYPE_AUDIO] >= 0 ?

                                 st_index[AVMEDIA_TYPE_AUDIO] :

                                 st_index[AVMEDIA_TYPE_VIDEO]),

                                NULL, 0);

    if (show_status) {

        av_dump_format(ic, 0, is->filename, 0);

    }



    /* open the streams */

    if (st_index[AVMEDIA_TYPE_AUDIO] >= 0) {

        stream_component_open(is, st_index[AVMEDIA_TYPE_AUDIO]);

    }



    ret=-1;

    if (st_index[AVMEDIA_TYPE_VIDEO] >= 0) {

        ret= stream_component_open(is, st_index[AVMEDIA_TYPE_VIDEO]);

    }

    is->refresh_tid = SDL_CreateThread(refresh_thread, is);

    if(ret<0) {

        if (!display_disable)

            is->show_audio = 2;

    }



    if (st_index[AVMEDIA_TYPE_SUBTITLE] >= 0) {

        stream_component_open(is, st_index[AVMEDIA_TYPE_SUBTITLE]);

    }



    if (is->video_stream < 0 && is->audio_stream < 0) {

        fprintf(stderr, "%s: could not open codecs\n", is->filename);

        ret = -1;

        goto fail;

    }



    for(;;) {

        if (is->abort_request)

            break;

        if (is->paused != is->last_paused) {

            is->last_paused = is->paused;

            if (is->paused)

                is->read_pause_return= av_read_pause(ic);

            else

                av_read_play(ic);

        }

#if CONFIG_RTSP_DEMUXER

        if (is->paused && !strcmp(ic->iformat->name, "rtsp")) {

            /* wait 10 ms to avoid trying to get another packet */

            /* XXX: horrible */

            SDL_Delay(10);

            continue;

        }

#endif

        if (is->seek_req) {

            int64_t seek_target= is->seek_pos;

            int64_t seek_min= is->seek_rel > 0 ? seek_target - is->seek_rel + 2: INT64_MIN;

            int64_t seek_max= is->seek_rel < 0 ? seek_target - is->seek_rel - 2: INT64_MAX;

//FIXME the +-2 is due to rounding being not done in the correct direction in generation

//      of the seek_pos/seek_rel variables



            ret = avformat_seek_file(is->ic, -1, seek_min, seek_target, seek_max, is->seek_flags);

            if (ret < 0) {

                fprintf(stderr, "%s: error while seeking\n", is->ic->filename);

            }else{

                if (is->audio_stream >= 0) {

                    packet_queue_flush(&is->audioq);

                    packet_queue_put(&is->audioq, &flush_pkt);

                }

                if (is->subtitle_stream >= 0) {

                    packet_queue_flush(&is->subtitleq);

                    packet_queue_put(&is->subtitleq, &flush_pkt);

                }

                if (is->video_stream >= 0) {

                    packet_queue_flush(&is->videoq);

                    packet_queue_put(&is->videoq, &flush_pkt);

                }

            }

            is->seek_req = 0;

            eof= 0;

        }



        /* if the queue are full, no need to read more */

        if (   is->audioq.size + is->videoq.size + is->subtitleq.size > MAX_QUEUE_SIZE

            || (   (is->audioq   .size  > MIN_AUDIOQ_SIZE || is->audio_stream<0)

                && (is->videoq   .nb_packets > MIN_FRAMES || is->video_stream<0)

                && (is->subtitleq.nb_packets > MIN_FRAMES || is->subtitle_stream<0))) {

            /* wait 10 ms */

            SDL_Delay(10);

            continue;

        }

        if(eof) {

            if(is->video_stream >= 0){

                av_init_packet(pkt);

                pkt->data=NULL;

                pkt->size=0;

                pkt->stream_index= is->video_stream;

                packet_queue_put(&is->videoq, pkt);

            }

            SDL_Delay(10);

            if(is->audioq.size + is->videoq.size + is->subtitleq.size ==0){

                if(loop!=1 && (!loop || --loop)){

                    stream_seek(cur_stream, start_time != AV_NOPTS_VALUE ? start_time : 0, 0, 0);

                }else if(autoexit){

                    ret=AVERROR_EOF;

                    goto fail;

                }

            }

            continue;

        }

        ret = av_read_frame(ic, pkt);

        if (ret < 0) {

            if (ret == AVERROR_EOF || ic->pb->eof_reached)

                eof=1;

            if (ic->pb->error)

                break;

            SDL_Delay(100); /* wait for user event */

            continue;

        }

        /* check if packet is in play range specified by user, then queue, otherwise discard */

        pkt_in_play_range = duration == AV_NOPTS_VALUE ||

                (pkt->pts - ic->streams[pkt->stream_index]->start_time) *

                av_q2d(ic->streams[pkt->stream_index]->time_base) -

                (double)(start_time != AV_NOPTS_VALUE ? start_time : 0)/1000000

                <= ((double)duration/1000000);

        if (pkt->stream_index == is->audio_stream && pkt_in_play_range) {

            packet_queue_put(&is->audioq, pkt);

        } else if (pkt->stream_index == is->video_stream && pkt_in_play_range) {

            packet_queue_put(&is->videoq, pkt);

        } else if (pkt->stream_index == is->subtitle_stream && pkt_in_play_range) {

            packet_queue_put(&is->subtitleq, pkt);

        } else {

            av_free_packet(pkt);

        }

    }

    /* wait until the end */

    while (!is->abort_request) {

        SDL_Delay(100);

    }



    ret = 0;

 fail:

    /* disable interrupting */

    global_video_state = NULL;



    /* close each stream */

    if (is->audio_stream >= 0)

        stream_component_close(is, is->audio_stream);

    if (is->video_stream >= 0)

        stream_component_close(is, is->video_stream);

    if (is->subtitle_stream >= 0)

        stream_component_close(is, is->subtitle_stream);

    if (is->ic) {

        av_close_input_file(is->ic);

        is->ic = NULL; /* safety */

    }

    url_set_interrupt_cb(NULL);



    if (ret != 0) {

        SDL_Event event;



        event.type = FF_QUIT_EVENT;

        event.user.data1 = is;

        SDL_PushEvent(&event);

    }

    return 0;

}
