static int lavfi_read_packet(AVFormatContext *avctx, AVPacket *pkt)

{

    LavfiContext *lavfi = avctx->priv_data;

    double min_pts = DBL_MAX;

    int stream_idx, min_pts_sink_idx = 0;

    AVFrame *frame = lavfi->decoded_frame;

    AVPicture pict;

    AVDictionary *frame_metadata;

    int ret, i;

    int size = 0;



    if (lavfi->subcc_packet.size) {

        *pkt = lavfi->subcc_packet;

        av_init_packet(&lavfi->subcc_packet);

        lavfi->subcc_packet.size = 0;

        lavfi->subcc_packet.data = NULL;

        return pkt->size;

    }



    /* iterate through all the graph sinks. Select the sink with the

     * minimum PTS */

    for (i = 0; i < lavfi->nb_sinks; i++) {

        AVRational tb = lavfi->sinks[i]->inputs[0]->time_base;

        double d;

        int ret;



        if (lavfi->sink_eof[i])

            continue;



        ret = av_buffersink_get_frame_flags(lavfi->sinks[i], frame,

                                            AV_BUFFERSINK_FLAG_PEEK);

        if (ret == AVERROR_EOF) {

            av_dlog(avctx, "EOF sink_idx:%d\n", i);

            lavfi->sink_eof[i] = 1;

            continue;

        } else if (ret < 0)

            return ret;

        d = av_rescale_q_rnd(frame->pts, tb, AV_TIME_BASE_Q, AV_ROUND_NEAR_INF|AV_ROUND_PASS_MINMAX);

        av_dlog(avctx, "sink_idx:%d time:%f\n", i, d);

        av_frame_unref(frame);



        if (d < min_pts) {

            min_pts = d;

            min_pts_sink_idx = i;

        }

    }

    if (min_pts == DBL_MAX)

        return AVERROR_EOF;



    av_dlog(avctx, "min_pts_sink_idx:%i\n", min_pts_sink_idx);



    av_buffersink_get_frame_flags(lavfi->sinks[min_pts_sink_idx], frame, 0);

    stream_idx = lavfi->sink_stream_map[min_pts_sink_idx];



    if (frame->width /* FIXME best way of testing a video */) {

        size = avpicture_get_size(frame->format, frame->width, frame->height);

        if ((ret = av_new_packet(pkt, size)) < 0)

            return ret;



        memcpy(pict.data,     frame->data,     4*sizeof(frame->data[0]));

        memcpy(pict.linesize, frame->linesize, 4*sizeof(frame->linesize[0]));



        avpicture_layout(&pict, frame->format, frame->width, frame->height,

                         pkt->data, size);

    } else if (av_frame_get_channels(frame) /* FIXME test audio */) {

        size = frame->nb_samples * av_get_bytes_per_sample(frame->format) *

                                   av_frame_get_channels(frame);

        if ((ret = av_new_packet(pkt, size)) < 0)

            return ret;

        memcpy(pkt->data, frame->data[0], size);

    }



    frame_metadata = av_frame_get_metadata(frame);

    if (frame_metadata) {

        uint8_t *metadata;

        AVDictionaryEntry *e = NULL;

        AVBPrint meta_buf;



        av_bprint_init(&meta_buf, 0, AV_BPRINT_SIZE_UNLIMITED);

        while ((e = av_dict_get(frame_metadata, "", e, AV_DICT_IGNORE_SUFFIX))) {

            av_bprintf(&meta_buf, "%s", e->key);

            av_bprint_chars(&meta_buf, '\0', 1);

            av_bprintf(&meta_buf, "%s", e->value);

            av_bprint_chars(&meta_buf, '\0', 1);

        }

        if (!av_bprint_is_complete(&meta_buf) ||

            !(metadata = av_packet_new_side_data(pkt, AV_PKT_DATA_STRINGS_METADATA,

                                                 meta_buf.len))) {

            av_bprint_finalize(&meta_buf, NULL);

            return AVERROR(ENOMEM);

        }

        memcpy(metadata, meta_buf.str, meta_buf.len);

        av_bprint_finalize(&meta_buf, NULL);

    }



    if ((ret = create_subcc_packet(avctx, frame, min_pts_sink_idx)) < 0) {

        av_frame_unref(frame);

        av_packet_unref(pkt);

        return ret;

    }



    pkt->stream_index = stream_idx;

    pkt->pts = frame->pts;

    pkt->pos = av_frame_get_pkt_pos(frame);

    pkt->size = size;

    av_frame_unref(frame);

    return size;

}
