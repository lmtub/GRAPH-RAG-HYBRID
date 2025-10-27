static CharDriverState *qemu_chr_open_socket(QemuOpts *opts)

{

    CharDriverState *chr = NULL;

    TCPCharDriver *s = NULL;

    int fd = -1;

    int is_listen;

    int is_waitconnect;

    int do_nodelay;

    int is_unix;

    int is_telnet;



    is_listen      = qemu_opt_get_bool(opts, "server", 0);

    is_waitconnect = qemu_opt_get_bool(opts, "wait", 1);

    is_telnet      = qemu_opt_get_bool(opts, "telnet", 0);

    do_nodelay     = !qemu_opt_get_bool(opts, "delay", 1);

    is_unix        = qemu_opt_get(opts, "path") != NULL;

    if (!is_listen)

        is_waitconnect = 0;



    chr = g_malloc0(sizeof(CharDriverState));

    s = g_malloc0(sizeof(TCPCharDriver));



    if (is_unix) {

        if (is_listen) {

            fd = unix_listen_opts(opts);

        } else {

            fd = unix_connect_opts(opts);

        }

    } else {

        if (is_listen) {

            fd = inet_listen_opts(opts, 0, NULL);

        } else {

            fd = inet_connect_opts(opts, NULL);

        }

    }

    if (fd < 0) {

        goto fail;

    }



    if (!is_waitconnect)

        socket_set_nonblock(fd);



    s->connected = 0;

    s->fd = -1;

    s->listen_fd = -1;

    s->msgfd = -1;

    s->is_unix = is_unix;

    s->do_nodelay = do_nodelay && !is_unix;



    chr->opaque = s;

    chr->chr_write = tcp_chr_write;

    chr->chr_close = tcp_chr_close;

    chr->get_msgfd = tcp_get_msgfd;

    chr->chr_add_client = tcp_chr_add_client;



    if (is_listen) {

        s->listen_fd = fd;

        qemu_set_fd_handler2(s->listen_fd, NULL, tcp_chr_accept, NULL, chr);

        if (is_telnet)

            s->do_telnetopt = 1;



    } else {

        s->connected = 1;

        s->fd = fd;

        socket_set_nodelay(fd);

        tcp_chr_connect(chr);

    }



    /* for "info chardev" monitor command */

    chr->filename = g_malloc(256);

    if (is_unix) {

        snprintf(chr->filename, 256, "unix:%s%s",

                 qemu_opt_get(opts, "path"),

                 qemu_opt_get_bool(opts, "server", 0) ? ",server" : "");

    } else if (is_telnet) {

        snprintf(chr->filename, 256, "telnet:%s:%s%s",

                 qemu_opt_get(opts, "host"), qemu_opt_get(opts, "port"),

                 qemu_opt_get_bool(opts, "server", 0) ? ",server" : "");

    } else {

        snprintf(chr->filename, 256, "tcp:%s:%s%s",

                 qemu_opt_get(opts, "host"), qemu_opt_get(opts, "port"),

                 qemu_opt_get_bool(opts, "server", 0) ? ",server" : "");

    }



    if (is_listen && is_waitconnect) {

        printf("QEMU waiting for connection on: %s\n",

               chr->filename);

        tcp_chr_accept(chr);

        socket_set_nonblock(s->listen_fd);

    }

    return chr;



 fail:

    if (fd >= 0)

        closesocket(fd);

    g_free(s);

    g_free(chr);

    return NULL;

}
