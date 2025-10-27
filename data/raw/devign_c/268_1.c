void ff_rtsp_undo_setup(AVFormatContext *s)

{

    RTSPState *rt = s->priv_data;

    int i;



    for (i = 0; i < rt->nb_rtsp_streams; i++) {

        RTSPStream *rtsp_st = rt->rtsp_streams[i];

        if (!rtsp_st)

            continue;

        if (rtsp_st->transport_priv) {

            if (s->oformat) {

                AVFormatContext *rtpctx = rtsp_st->transport_priv;

                av_write_trailer(rtpctx);

                if (rt->lower_transport == RTSP_LOWER_TRANSPORT_TCP) {

                    uint8_t *ptr;

                    url_close_dyn_buf(rtpctx->pb, &ptr);

                    av_free(ptr);

                } else {

                    url_fclose(rtpctx->pb);

                }

                av_metadata_free(&rtpctx->streams[0]->metadata);

                av_metadata_free(&rtpctx->metadata);


                av_free(rtpctx->streams[0]);

                av_free(rtpctx);

            } else if (rt->transport == RTSP_TRANSPORT_RDT && CONFIG_RTPDEC)

                ff_rdt_parse_close(rtsp_st->transport_priv);

            else if (CONFIG_RTPDEC)

                rtp_parse_close(rtsp_st->transport_priv);

        }

        rtsp_st->transport_priv = NULL;

        if (rtsp_st->rtp_handle)

            url_close(rtsp_st->rtp_handle);

        rtsp_st->rtp_handle = NULL;

    }

}