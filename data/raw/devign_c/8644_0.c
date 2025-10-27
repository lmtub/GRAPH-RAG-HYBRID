static int tcp_chr_add_client(CharDriverState *chr, int fd)

{

    TCPCharDriver *s = chr->opaque;

    if (s->fd != -1)

	return -1;



    qemu_set_nonblock(fd);

    if (s->do_nodelay)

        socket_set_nodelay(fd);

    s->fd = fd;

    s->chan = io_channel_from_socket(fd);

    g_source_remove(s->listen_tag);

    s->listen_tag = 0;

    tcp_chr_connect(chr);



    return 0;

}
