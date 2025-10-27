static int tcp_read_packet(AVFormatContext *s, RTSPStream **prtsp_st,

                           uint8_t *buf, int buf_size)

{

    RTSPState *rt = s->priv_data;

    int id, len, i, ret;

    RTSPStream *rtsp_st;



#ifdef DEBUG_RTP_TCP

    dprintf(s, "tcp_read_packet:\n");

#endif

 redo:

    for(;;) {

        RTSPMessageHeader reply;



        ret = rtsp_read_reply(s, &reply, NULL, 1);

        if (ret == -1)

            return -1;

        if (ret == 1) /* received '$' */

            break;

        /* XXX: parse message */

        if (rt->state != RTSP_STATE_PLAYING)

            return 0;

    }

    ret = url_read_complete(rt->rtsp_hd, buf, 3);

    if (ret != 3)

        return -1;

    id = buf[0];

    len = AV_RB16(buf + 1);

#ifdef DEBUG_RTP_TCP

    dprintf(s, "id=%d len=%d\n", id, len);

#endif

    if (len > buf_size || len < 12)

        goto redo;

    /* get the data */

    ret = url_read_complete(rt->rtsp_hd, buf, len);

    if (ret != len)

        return -1;

    if (rt->transport == RTSP_TRANSPORT_RDT &&

        ff_rdt_parse_header(buf, len, &id, NULL, NULL, NULL, NULL) < 0)

        return -1;



    /* find the matching stream */

    for(i = 0; i < rt->nb_rtsp_streams; i++) {

        rtsp_st = rt->rtsp_streams[i];

        if (id >= rtsp_st->interleaved_min &&

            id <= rtsp_st->interleaved_max)

            goto found;

    }

    goto redo;

 found:

    *prtsp_st = rtsp_st;

    return len;

}
