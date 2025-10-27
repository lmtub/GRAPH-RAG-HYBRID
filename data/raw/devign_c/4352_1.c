static int qemu_chr_open_udp(QemuOpts *opts, CharDriverState **_chr)

{

    CharDriverState *chr = NULL;

    NetCharDriver *s = NULL;

    int fd = -1;

    int ret;



    chr = g_malloc0(sizeof(CharDriverState));

    s = g_malloc0(sizeof(NetCharDriver));



    fd = inet_dgram_opts(opts);

    if (fd < 0) {

        fprintf(stderr, "inet_dgram_opts failed\n");

        ret = -errno;

        goto return_err;

    }



    s->fd = fd;

    s->bufcnt = 0;

    s->bufptr = 0;

    chr->opaque = s;

    chr->chr_write = udp_chr_write;

    chr->chr_update_read_handler = udp_chr_update_read_handler;

    chr->chr_close = udp_chr_close;



    *_chr = chr;

    return 0;



return_err:

    g_free(chr);

    g_free(s);

    if (fd >= 0) {

        closesocket(fd);

    }

    return ret;

}
