static void udp_chr_close(CharDriverState *chr)

{

    NetCharDriver *s = chr->opaque;

    if (s->tag) {

        g_source_remove(s->tag);

        s->tag = 0;

    }

    if (s->chan) {

        g_io_channel_unref(s->chan);

        closesocket(s->fd);

    }

    g_free(s);

    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

}
