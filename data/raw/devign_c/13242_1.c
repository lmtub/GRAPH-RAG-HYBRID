static CharDriverState *qemu_chr_open_stdio(QemuOpts *opts)

{

    CharDriverState *chr;



    if (stdio_nb_clients >= STDIO_MAX_CLIENTS) {







    if (stdio_nb_clients == 0) {

        old_fd0_flags = fcntl(0, F_GETFL);

        tcgetattr (0, &oldtty);

        fcntl(0, F_SETFL, O_NONBLOCK);

        atexit(term_exit);




    chr = qemu_chr_open_fd(0, 1);

    chr->chr_close = qemu_chr_close_stdio;

    chr->chr_set_echo = qemu_chr_set_echo_stdio;

    qemu_set_fd_handler2(0, stdio_read_poll, stdio_read, NULL, chr);

    stdio_nb_clients++;

    stdio_allow_signal = qemu_opt_get_bool(opts, "signal",

                                           display_type != DT_NOGRAPHIC);

    qemu_chr_fe_set_echo(chr, false);



    return chr;
