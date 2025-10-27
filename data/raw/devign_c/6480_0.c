static int read_frame_internal(AVFormatContext *s, AVPacket *pkt)

{

    int ret = 0, i, got_packet = 0;

    AVDictionary *metadata = NULL;



    av_init_packet(pkt);



    while (!got_packet && !s->parse_queue) {

        AVStream *st;

        AVPacket cur_pkt;



        /* read next packet */

        ret = ff_read_packet(s, &cur_pkt);

        if (ret < 0) {

            if (ret == AVERROR(EAGAIN))

                return ret;

            /* flush the parsers */

            for (i = 0; i < s->nb_streams; i++) {

                st = s->streams[i];

                if (st->parser && st->need_parsing)

                    parse_packet(s, NULL, st->index);

            }

            /* all remaining packets are now in parse_queue =>

             * really terminate parsing */

            break;

        }

        ret = 0;

        st  = s->streams[cur_pkt.stream_index];



        if (cur_pkt.pts != AV_NOPTS_VALUE &&

            cur_pkt.dts != AV_NOPTS_VALUE &&

            cur_pkt.pts < cur_pkt.dts) {

            av_log(s, AV_LOG_WARNING,

                   "Invalid timestamps stream=%d, pts=%s, dts=%s, size=%d\n",

                   cur_pkt.stream_index,

                   av_ts2str(cur_pkt.pts),

                   av_ts2str(cur_pkt.dts),

                   cur_pkt.size);

        }

        if (s->debug & FF_FDEBUG_TS)

            av_log(s, AV_LOG_DEBUG,

                   "ff_read_packet stream=%d, pts=%s, dts=%s, size=%d, duration=%d, flags=%d\n",

                   cur_pkt.stream_index,

                   av_ts2str(cur_pkt.pts),

                   av_ts2str(cur_pkt.dts),

                   cur_pkt.size, cur_pkt.duration, cur_pkt.flags);



        if (st->need_parsing && !st->parser && !(s->flags & AVFMT_FLAG_NOPARSE)) {

            st->parser = av_parser_init(st->codec->codec_id);

            if (!st->parser) {

                av_log(s, AV_LOG_VERBOSE, "parser not found for codec "

                       "%s, packets or times may be invalid.\n",

                       avcodec_get_name(st->codec->codec_id));

                /* no parser available: just output the raw packets */

                st->need_parsing = AVSTREAM_PARSE_NONE;

            } else if (st->need_parsing == AVSTREAM_PARSE_HEADERS)

                st->parser->flags |= PARSER_FLAG_COMPLETE_FRAMES;

            else if (st->need_parsing == AVSTREAM_PARSE_FULL_ONCE)

                st->parser->flags |= PARSER_FLAG_ONCE;

            else if (st->need_parsing == AVSTREAM_PARSE_FULL_RAW)

                st->parser->flags |= PARSER_FLAG_USE_CODEC_TS;

        }



        if (!st->need_parsing || !st->parser) {

            /* no parsing needed: we just output the packet as is */

            *pkt = cur_pkt;

            compute_pkt_fields(s, st, NULL, pkt);

            if ((s->iformat->flags & AVFMT_GENERIC_INDEX) &&

                (pkt->flags & AV_PKT_FLAG_KEY) && pkt->dts != AV_NOPTS_VALUE) {

                ff_reduce_index(s, st->index);

                av_add_index_entry(st, pkt->pos, pkt->dts,

                                   0, 0, AVINDEX_KEYFRAME);

            }

            got_packet = 1;

        } else if (st->discard < AVDISCARD_ALL) {

            if ((ret = parse_packet(s, &cur_pkt, cur_pkt.stream_index)) < 0)

                return ret;

        } else {

            /* free packet */

            av_free_packet(&cur_pkt);

        }

        if (pkt->flags & AV_PKT_FLAG_KEY)

            st->skip_to_keyframe = 0;

        if (st->skip_to_keyframe) {

            av_free_packet(&cur_pkt);

            if (got_packet) {

                *pkt = cur_pkt;

            }

            got_packet = 0;

        }

    }



    if (!got_packet && s->parse_queue)

        ret = read_from_packet_buffer(&s->parse_queue, &s->parse_queue_end, pkt);



    if (ret >= 0) {

        AVStream *st = s->streams[pkt->stream_index];

        int discard_padding = 0;

        if (st->end_discard_sample && pkt->pts != AV_NOPTS_VALUE) {

            int64_t pts = pkt->pts - (is_relative(pkt->pts) ? RELATIVE_TS_BASE : 0);

            int64_t sample = ts_to_samples(st, pts);

            int duration = ts_to_samples(st, pkt->duration);

            int64_t end_sample = sample + duration;

            if (duration > 0 && end_sample >= st->end_discard_sample)

                discard_padding = FFMIN(end_sample - st->end_discard_sample, duration);

        }

        if (st->skip_samples || discard_padding) {

            uint8_t *p = av_packet_new_side_data(pkt, AV_PKT_DATA_SKIP_SAMPLES, 10);

            if (p) {

                AV_WL32(p, st->skip_samples);

                AV_WL32(p + 4, discard_padding);

                av_log(s, AV_LOG_DEBUG, "demuxer injecting skip %d\n", st->skip_samples);

            }

            st->skip_samples = 0;

        }



        if (st->inject_global_side_data) {

            for (i = 0; i < st->nb_side_data; i++) {

                AVPacketSideData *src_sd = &st->side_data[i];

                uint8_t *dst_data;



                if (av_packet_get_side_data(pkt, src_sd->type, NULL))

                    continue;



                dst_data = av_packet_new_side_data(pkt, src_sd->type, src_sd->size);

                if (!dst_data) {

                    av_log(s, AV_LOG_WARNING, "Could not inject global side data\n");

                    continue;

                }



                memcpy(dst_data, src_sd->data, src_sd->size);

            }

            st->inject_global_side_data = 0;

        }



        if (!(s->flags & AVFMT_FLAG_KEEP_SIDE_DATA))

            av_packet_merge_side_data(pkt);

    }



    av_opt_get_dict_val(s, "metadata", AV_OPT_SEARCH_CHILDREN, &metadata);

    if (metadata) {

        s->event_flags |= AVFMT_EVENT_FLAG_METADATA_UPDATED;

        av_dict_copy(&s->metadata, metadata, 0);

        av_dict_free(&metadata);

        av_opt_set_dict_val(s, "metadata", NULL, AV_OPT_SEARCH_CHILDREN);

    }



    if (s->debug & FF_FDEBUG_TS)

        av_log(s, AV_LOG_DEBUG,

               "read_frame_internal stream=%d, pts=%s, dts=%s, "

               "size=%d, duration=%d, flags=%d\n",

               pkt->stream_index,

               av_ts2str(pkt->pts),

               av_ts2str(pkt->dts),

               pkt->size, pkt->duration, pkt->flags);



    return ret;

}
