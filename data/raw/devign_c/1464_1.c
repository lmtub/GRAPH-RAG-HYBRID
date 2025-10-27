static int parse_packet(AVFormatContext *s, AVPacket *pkt, int stream_index)

{

    AVPacket out_pkt = { 0 }, flush_pkt = { 0 };

    AVStream *st = s->streams[stream_index];

    uint8_t *data = pkt ? pkt->data : NULL;

    int size      = pkt ? pkt->size : 0;

    int ret = 0, got_output = 0;



    if (!pkt) {

        av_init_packet(&flush_pkt);

        pkt        = &flush_pkt;

        got_output = 1;

    } else if (!size && st->parser->flags & PARSER_FLAG_COMPLETE_FRAMES) {

        // preserve 0-size sync packets

        compute_pkt_fields(s, st, st->parser, pkt, AV_NOPTS_VALUE, AV_NOPTS_VALUE);

    }



    while (size > 0 || (pkt == &flush_pkt && got_output)) {

        int len;

        int64_t next_pts = pkt->pts;

        int64_t next_dts = pkt->dts;



        av_init_packet(&out_pkt);

        len = av_parser_parse2(st->parser, st->codec,

                               &out_pkt.data, &out_pkt.size, data, size,

                               pkt->pts, pkt->dts, pkt->pos);



        pkt->pts = pkt->dts = AV_NOPTS_VALUE;

        pkt->pos = -1;

        /* increment read pointer */

        data += len;

        size -= len;



        got_output = !!out_pkt.size;



        if (!out_pkt.size)

            continue;



        if (pkt->side_data) {

            out_pkt.side_data       = pkt->side_data;

            out_pkt.side_data_elems = pkt->side_data_elems;

            pkt->side_data          = NULL;

            pkt->side_data_elems    = 0;

        }



        /* set the duration */

        out_pkt.duration = (st->parser->flags & PARSER_FLAG_COMPLETE_FRAMES) ? pkt->duration : 0;

        if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (st->codec->sample_rate > 0) {

                out_pkt.duration =

                    av_rescale_q_rnd(st->parser->duration,

                                     (AVRational) { 1, st->codec->sample_rate },

                                     st->time_base,

                                     AV_ROUND_DOWN);

            }

        }



        out_pkt.stream_index = st->index;

        out_pkt.pts          = st->parser->pts;

        out_pkt.dts          = st->parser->dts;

        out_pkt.pos          = st->parser->pos;



        if (st->need_parsing == AVSTREAM_PARSE_FULL_RAW)

            out_pkt.pos = st->parser->frame_offset;



        if (st->parser->key_frame == 1 ||

            (st->parser->key_frame == -1 &&

             st->parser->pict_type == AV_PICTURE_TYPE_I))

            out_pkt.flags |= AV_PKT_FLAG_KEY;



        if (st->parser->key_frame == -1 && st->parser->pict_type ==AV_PICTURE_TYPE_NONE && (pkt->flags&AV_PKT_FLAG_KEY))

            out_pkt.flags |= AV_PKT_FLAG_KEY;



        compute_pkt_fields(s, st, st->parser, &out_pkt, next_dts, next_pts);



        if ((ret = add_to_pktbuf(&s->internal->parse_queue, &out_pkt,

                                 &s->internal->parse_queue_end,

                                 1))) {

            av_packet_unref(&out_pkt);

            goto fail;

        }

    }



    /* end of the stream => close and free the parser */

    if (pkt == &flush_pkt) {

        av_parser_close(st->parser);

        st->parser = NULL;

    }



fail:

    av_packet_unref(pkt);

    return ret;

}
