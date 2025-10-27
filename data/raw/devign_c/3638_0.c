rtsp_open_transport_ctx(AVFormatContext *s, RTSPStream *rtsp_st)

{

    RTSPState *rt = s->priv_data;

    AVStream *st = NULL;



    /* open the RTP context */

    if (rtsp_st->stream_index >= 0)

        st = s->streams[rtsp_st->stream_index];

    if (!st)

        s->ctx_flags |= AVFMTCTX_NOHEADER;



    if (rt->transport == RTSP_TRANSPORT_RDT)

        rtsp_st->transport_priv = ff_rdt_parse_open(s, st->index,

                                            rtsp_st->dynamic_protocol_context,

                                            rtsp_st->dynamic_handler);

    else

        rtsp_st->transport_priv = rtp_parse_open(s, st, rtsp_st->rtp_handle,

                                         rtsp_st->sdp_payload_type,

                                         &rtsp_st->rtp_payload_data);



    if (!rtsp_st->transport_priv) {

         return AVERROR(ENOMEM);

    } else if (rt->transport != RTSP_TRANSPORT_RDT) {

        if(rtsp_st->dynamic_handler) {

            rtp_parse_set_dynamic_protocol(rtsp_st->transport_priv,

                                           rtsp_st->dynamic_protocol_context,

                                           rtsp_st->dynamic_handler);

        }

    }



    return 0;

}
