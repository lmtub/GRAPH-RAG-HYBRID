static void tcp_chr_disconnect(CharDriverState *chr)

{

    TCPCharDriver *s = chr->opaque;



    s->connected = 0;

    if (s->listen_chan) {

        s->listen_tag = g_io_add_watch(s->listen_chan, G_IO_IN,

                                       tcp_chr_accept, chr);

    }

    remove_fd_in_watch(chr);

    g_io_channel_unref(s->chan);

    s->chan = NULL;

    closesocket(s->fd);

    s->fd = -1;

    SocketAddress_to_str(chr->filename, CHR_MAX_FILENAME_SIZE,

                         "disconnected:", s->addr, s->is_listen, s->is_telnet);

    qemu_chr_be_event(chr, CHR_EVENT_CLOSED);

    if (s->reconnect_time) {

        qemu_chr_socket_restart_timer(chr);

    }

}
