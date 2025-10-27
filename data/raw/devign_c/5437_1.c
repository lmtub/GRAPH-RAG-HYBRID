int ff_wms_parse_sdp_a_line(AVFormatContext *s, const char *p)

{

    int ret = 0;

    if (av_strstart(p, "pgmpu:data:application/vnd.ms.wms-hdr.asfv1;base64,", &p)) {

        AVIOContext pb;

        RTSPState *rt = s->priv_data;

        AVDictionary *opts = NULL;

        int len = strlen(p) * 6 / 8;

        char *buf = av_mallocz(len);

        AVInputFormat *iformat;



        if (!buf)

            return AVERROR(ENOMEM);

        av_base64_decode(buf, p, len);



        if (rtp_asf_fix_header(buf, len) < 0)

            av_log(s, AV_LOG_ERROR,

                   "Failed to fix invalid RTSP-MS/ASF min_pktsize\n");

        init_packetizer(&pb, buf, len);

        if (rt->asf_ctx) {

            avformat_close_input(&rt->asf_ctx);

        }



        if (!(iformat = av_find_input_format("asf")))

            return AVERROR_DEMUXER_NOT_FOUND;



        rt->asf_ctx = avformat_alloc_context();

        if (!rt->asf_ctx) {

            av_free(buf);

            return AVERROR(ENOMEM);

        }

        rt->asf_ctx->pb      = &pb;

        av_dict_set(&opts, "no_resync_search", "1", 0);



        if ((ret = ff_copy_whiteblacklists(rt->asf_ctx, s)) < 0) {

            av_dict_free(&opts);

            return ret;

        }



        ret = avformat_open_input(&rt->asf_ctx, "", iformat, &opts);

        av_dict_free(&opts);

        if (ret < 0) {

            av_free(buf);

            return ret;

        }

        av_dict_copy(&s->metadata, rt->asf_ctx->metadata, 0);

        rt->asf_pb_pos = avio_tell(&pb);

        av_free(buf);

        rt->asf_ctx->pb = NULL;

    }

    return ret;

}
