CharDriverState *qemu_chr_open(const char *filename)

{

    const char *p;



    if (!strcmp(filename, "vc")) {

        return text_console_init(&display_state);

    } else if (!strcmp(filename, "null")) {

        return qemu_chr_open_null();

    } else 

    if (strstart(filename, "tcp:", &p)) {

        return qemu_chr_open_tcp(p, 0, 0);

    } else

    if (strstart(filename, "telnet:", &p)) {

        return qemu_chr_open_tcp(p, 1, 0);

    } else

    if (strstart(filename, "udp:", &p)) {

        return qemu_chr_open_udp(p);

    } else

    if (strstart(filename, "mon:", &p)) {

        CharDriverState *drv = qemu_chr_open(p);

        if (drv) {

            drv = qemu_chr_open_mux(drv);

            monitor_init(drv, !nographic);

            return drv;

        }

        printf("Unable to open driver: %s\n", p);

        return 0;

    } else

#ifndef _WIN32

    if (strstart(filename, "unix:", &p)) {

	return qemu_chr_open_tcp(p, 0, 1);

    } else if (strstart(filename, "file:", &p)) {

        return qemu_chr_open_file_out(p);

    } else if (strstart(filename, "pipe:", &p)) {

        return qemu_chr_open_pipe(p);

    } else if (!strcmp(filename, "pty")) {

        return qemu_chr_open_pty();

    } else if (!strcmp(filename, "stdio")) {

        return qemu_chr_open_stdio();

    } else 

#if defined(__linux__)

    if (strstart(filename, "/dev/parport", NULL)) {

        return qemu_chr_open_pp(filename);

    } else 

#endif

    if (strstart(filename, "/dev/", NULL)) {

        return qemu_chr_open_tty(filename);

    } else 

#else /* !_WIN32 */

    if (strstart(filename, "COM", NULL)) {

        return qemu_chr_open_win(filename);

    } else

    if (strstart(filename, "pipe:", &p)) {

        return qemu_chr_open_win_pipe(p);

    } else

    if (strstart(filename, "con:", NULL)) {

        return qemu_chr_open_win_con(filename);

    } else

    if (strstart(filename, "file:", &p)) {

        return qemu_chr_open_win_file_out(p);

    }

#endif

    {

        return NULL;

    }

}
