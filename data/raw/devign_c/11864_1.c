static gboolean tcp_chr_read(GIOChannel *chan, GIOCondition cond, void *opaque)

{

    CharDriverState *chr = opaque;

    TCPCharDriver *s = chr->opaque;

    uint8_t buf[READ_BUF_LEN];

    int len, size;



    if (!s->connected || s->max_size <= 0) {

        return TRUE;

    }

    len = sizeof(buf);

    if (len > s->max_size)

        len = s->max_size;

    size = tcp_chr_recv(chr, (void *)buf, len);

    if (size == 0) {

        /* connection closed */

        s->connected = 0;

        if (s->listen_chan) {

            s->listen_tag = g_io_add_watch(s->listen_chan, G_IO_IN, tcp_chr_accept, chr);

        }

        if (s->tag) {

            g_source_remove(s->tag);

            s->tag = 0;

        }

        g_io_channel_unref(s->chan);

        s->chan = NULL;

        closesocket(s->fd);

        s->fd = -1;

        qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

    } else if (size > 0) {

        if (s->do_telnetopt)

            tcp_chr_process_IAC_bytes(chr, s, buf, &size);

        if (size > 0)

            qemu_chr_be_write(chr, buf, size);

    }



    return TRUE;

}
