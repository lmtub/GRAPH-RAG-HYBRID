static int sctp_write(URLContext *h, const uint8_t *buf, int size)

{

    SCTPContext *s = h->priv_data;

    int ret;



    if (!(h->flags & AVIO_FLAG_NONBLOCK)) {

        ret = sctp_wait_fd(s->fd, 1);

        if (ret < 0)

            return ret;

    }



    if (s->max_streams) {

        /*StreamId is introduced as a 2byte code into the stream*/

        struct sctp_sndrcvinfo info = { 0 };

        info.sinfo_stream           = AV_RB16(buf);

        if (info.sinfo_stream > s->max_streams)

            abort();

        ret = ff_sctp_send(s->fd, buf + 2, size - 2, &info, MSG_EOR);

    } else

        ret = send(s->fd, buf, size, 0);



    return ret < 0 ? ff_neterrno() : ret;

}
