static void rtsp_cmd_pause(HTTPContext *c, const char *url, RTSPHeader *h)

{

    HTTPContext *rtp_c;



    rtp_c = find_rtp_session_with_url(url, h->session_id);

    if (!rtp_c) {

        rtsp_reply_error(c, RTSP_STATUS_SESSION);

        return;

    }

    

    if (rtp_c->state != HTTPSTATE_SEND_DATA &&

        rtp_c->state != HTTPSTATE_WAIT_FEED) {

        rtsp_reply_error(c, RTSP_STATUS_STATE);

        return;

    }

    

    rtp_c->state = HTTPSTATE_READY;

    

    /* now everything is OK, so we can send the connection parameters */

    rtsp_reply_header(c, RTSP_STATUS_OK);

    /* session ID */

    url_fprintf(c->pb, "Session: %s\r\n", rtp_c->session_id);

    url_fprintf(c->pb, "\r\n");

}
