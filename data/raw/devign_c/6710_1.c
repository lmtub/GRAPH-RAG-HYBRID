static void tcp_chr_disconnect(CharDriverState *chr)

{

    TCPCharDriver *s = chr->opaque;



    if (!s->connected) {

        return;

    }



    s->connected = 0;

    if (s->listen_ioc) {

        s->listen_tag = qio_channel_add_watch(

            QIO_CHANNEL(s->listen_ioc), G_IO_IN, tcp_chr_accept, chr, NULL);

    }

    tcp_set_msgfds(chr, NULL, 0);

    remove_fd_in_watch(chr);

    object_unref(OBJECT(s->sioc));

    s->sioc = NULL;

    object_unref(OBJECT(s->ioc));

    s->ioc = NULL;

    g_free(chr->filename);

    chr->filename = SocketAddress_to_str("disconnected:", s->addr,

                                         s->is_listen, s->is_telnet);

    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

    if (s->reconnect_time) {

        qemu_chr_socket_restart_timer(chr);

    }

}
