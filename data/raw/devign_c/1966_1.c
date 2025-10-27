void ff_wms_parse_sdp_a_line(AVFormatContext *s, const char *p)

{

    if (av_strstart(p, "pgmpu:data:application/vnd.ms.wms-hdr.asfv1;base64,", &p)) {

        ByteIOContext pb;

        RTSPState *rt = s->priv_data;

        int len = strlen(p) * 6 / 8;

        char *buf = av_mallocz(len);

        av_base64_decode(buf, p, len);



        if (rtp_asf_fix_header(buf, len) < 0)

            av_log(s, AV_LOG_ERROR,

                   "Failed to fix invalid RTSP-MS/ASF min_pktsize\n");

        init_packetizer(&pb, buf, len);

        if (rt->asf_ctx) {

            av_close_input_stream(rt->asf_ctx);

            rt->asf_ctx = NULL;

        }

        av_open_input_stream(&rt->asf_ctx, &pb, "", &asf_demuxer, NULL);

        rt->asf_pb_pos = url_ftell(&pb);

        av_free(buf);

        rt->asf_ctx->pb = NULL;

    }

}
