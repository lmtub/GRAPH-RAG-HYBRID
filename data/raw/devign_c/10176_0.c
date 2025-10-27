static int sdp_read_header(AVFormatContext *s,

                           AVFormatParameters *ap)

{

    RTSPState *rt = s->priv_data;

    RTSPStream *rtsp_st;

    int size, i, err;

    char *content;

    char url[1024];



    /* read the whole sdp file */

    /* XXX: better loading */

    content = av_malloc(SDP_MAX_SIZE);

    size = get_buffer(s->pb, content, SDP_MAX_SIZE - 1);

    if (size <= 0) {

        av_free(content);

        return AVERROR_INVALIDDATA;

    }

    content[size] ='\0';



    sdp_parse(s, content);

    av_free(content);



    /* open each RTP stream */

    for(i=0;i<rt->nb_rtsp_streams;i++) {

        rtsp_st = rt->rtsp_streams[i];



        snprintf(url, sizeof(url), "rtp://%s:%d?localport=%d&ttl=%d",

                 inet_ntoa(rtsp_st->sdp_ip),

                 rtsp_st->sdp_port,

                 rtsp_st->sdp_port,

                 rtsp_st->sdp_ttl);

        if (url_open(&rtsp_st->rtp_handle, url, URL_RDWR) < 0) {

            err = AVERROR_INVALIDDATA;

            goto fail;

        }

        if ((err = rtsp_open_transport_ctx(s, rtsp_st)))

            goto fail;

    }

    return 0;

 fail:

    rtsp_close_streams(rt);

    return err;

}
