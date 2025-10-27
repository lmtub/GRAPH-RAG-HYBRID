int ff_rtsp_open_transport_ctx(AVFormatContext *s, RTSPStream *rtsp_st)

{

    RTSPState *rt = s->priv_data;

    AVStream *st = NULL;

    int reordering_queue_size = rt->reordering_queue_size;

    if (reordering_queue_size < 0) {

        if (rt->lower_transport == RTSP_LOWER_TRANSPORT_TCP || !s->max_delay)

            reordering_queue_size = 0;

        else

            reordering_queue_size = RTP_REORDER_QUEUE_DEFAULT_SIZE;

    }



    /* open the RTP context */

    if (rtsp_st->stream_index >= 0)

        st = s->streams[rtsp_st->stream_index];

    if (!st)

        s->ctx_flags |= AVFMTCTX_NOHEADER;



    if (CONFIG_RTSP_MUXER && s->oformat) {

        int ret = ff_rtp_chain_mux_open((AVFormatContext **)&rtsp_st->transport_priv,

                                        s, st, rtsp_st->rtp_handle,

                                        RTSP_TCP_MAX_PACKET_SIZE,

                                        rtsp_st->stream_index);

        /* Ownership of rtp_handle is passed to the rtp mux context */

        rtsp_st->rtp_handle = NULL;

        if (ret < 0)

            return ret;

        st->time_base = ((AVFormatContext*)rtsp_st->transport_priv)->streams[0]->time_base;

    } else if (rt->transport == RTSP_TRANSPORT_RAW) {

        return 0; // Don't need to open any parser here

    } else if (CONFIG_RTPDEC && rt->transport == RTSP_TRANSPORT_RDT)

        rtsp_st->transport_priv = ff_rdt_parse_open(s, st->index,

                                            rtsp_st->dynamic_protocol_context,

                                            rtsp_st->dynamic_handler);

    else if (CONFIG_RTPDEC)

        rtsp_st->transport_priv = ff_rtp_parse_open(s, st,

                                         rtsp_st->sdp_payload_type,

                                         reordering_queue_size);



    if (!rtsp_st->transport_priv) {

         return AVERROR(ENOMEM);

    } else if (CONFIG_RTPDEC && rt->transport == RTSP_TRANSPORT_RTP) {

        RTPDemuxContext *rtpctx = rtsp_st->transport_priv;

        rtpctx->ssrc = rtsp_st->ssrc;

        if (rtsp_st->dynamic_handler) {

            ff_rtp_parse_set_dynamic_protocol(rtsp_st->transport_priv,

                                              rtsp_st->dynamic_protocol_context,

                                              rtsp_st->dynamic_handler);

        }

        if (rtsp_st->crypto_suite[0])

            ff_rtp_parse_set_crypto(rtsp_st->transport_priv,

                                    rtsp_st->crypto_suite,

                                    rtsp_st->crypto_params);

    }



    return 0;

}
