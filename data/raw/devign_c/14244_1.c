static CharDriverState *qemu_chr_open_stdio(const char *id,

                                            ChardevBackend *backend,

                                            ChardevReturn *ret,

                                            Error **errp)

{

    ChardevStdio *opts = backend->u.stdio.data;

    CharDriverState *chr;

    struct sigaction act;

    ChardevCommon *common = qapi_ChardevStdio_base(opts);



    if (is_daemonized()) {

        error_setg(errp, "cannot use stdio with -daemonize");





    if (stdio_in_use) {

        error_setg(errp, "cannot use stdio by multiple character devices");





    stdio_in_use = true;

    old_fd0_flags = fcntl(0, F_GETFL);

    tcgetattr(0, &oldtty);

    qemu_set_nonblock(0);

    atexit(term_exit);



    memset(&act, 0, sizeof(act));

    act.sa_handler = term_stdio_handler;

    sigaction(SIGCONT, &act, NULL);



    chr = qemu_chr_open_fd(0, 1, common, errp);




    chr->chr_close = qemu_chr_close_stdio;

    chr->chr_set_echo = qemu_chr_set_echo_stdio;

    if (opts->has_signal) {

        stdio_allow_signal = opts->signal;


    qemu_chr_fe_set_echo(chr, false);



    return chr;
