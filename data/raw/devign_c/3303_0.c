static CharDriverState *qemu_chr_open_pty(QemuOpts *opts)

{

    CharDriverState *chr;

    PtyCharDriver *s;

    struct termios tty;

    const char *label;

    int master_fd, slave_fd, len;

#if defined(__OpenBSD__) || defined(__DragonFly__)

    char pty_name[PATH_MAX];

#define q_ptsname(x) pty_name

#else

    char *pty_name = NULL;

#define q_ptsname(x) ptsname(x)

#endif



    if (openpty(&master_fd, &slave_fd, pty_name, NULL, NULL) < 0) {

        return NULL;

    }



    /* Set raw attributes on the pty. */

    tcgetattr(slave_fd, &tty);

    cfmakeraw(&tty);

    tcsetattr(slave_fd, TCSAFLUSH, &tty);

    close(slave_fd);



    chr = g_malloc0(sizeof(CharDriverState));



    len = strlen(q_ptsname(master_fd)) + 5;

    chr->filename = g_malloc(len);

    snprintf(chr->filename, len, "pty:%s", q_ptsname(master_fd));

    qemu_opt_set(opts, "path", q_ptsname(master_fd));



    label = qemu_opts_id(opts);

    fprintf(stderr, "char device redirected to %s%s%s%s\n",

            q_ptsname(master_fd),

            label ? " (label " : "",

            label ? label      : "",

            label ? ")"        : "");



    s = g_malloc0(sizeof(PtyCharDriver));

    chr->opaque = s;

    chr->chr_write = pty_chr_write;

    chr->chr_update_read_handler = pty_chr_update_read_handler;

    chr->chr_close = pty_chr_close;

    chr->chr_add_watch = pty_chr_add_watch;



    s->fd = io_channel_from_fd(master_fd);

    s->timer_tag = 0;



    return chr;

}
