static void tcp_chr_connect(void *opaque)

{

    CharDriverState *chr = opaque;

    TCPCharDriver *s = chr->opaque;

    struct sockaddr_storage ss, ps;

    socklen_t ss_len = sizeof(ss), ps_len = sizeof(ps);



    memset(&ss, 0, ss_len);

    if (getsockname(s->fd, (struct sockaddr *) &ss, &ss_len) != 0) {

        snprintf(chr->filename, CHR_MAX_FILENAME_SIZE,

                 "Error in getsockname: %s\n", strerror(errno));

    } else if (getpeername(s->fd, (struct sockaddr *) &ps, &ps_len) != 0) {

        snprintf(chr->filename, CHR_MAX_FILENAME_SIZE,

                 "Error in getpeername: %s\n", strerror(errno));

    } else {

        sockaddr_to_str(chr->filename, CHR_MAX_FILENAME_SIZE,

                        &ss, ss_len, &ps, ps_len,

                        s->is_listen, s->is_telnet);

    }



    s->connected = 1;

    if (s->chan) {

        chr->fd_in_tag = io_add_watch_poll(s->chan, tcp_chr_read_poll,

                                           tcp_chr_read, chr);

    }

    qemu_chr_be_generic_open(chr);

}
