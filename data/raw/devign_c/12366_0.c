QemuOpts *qemu_chr_parse_compat(const char *label, const char *filename)

{

    char host[65], port[33], width[8], height[8];

    int pos;

    const char *p;

    QemuOpts *opts;

    Error *local_err = NULL;



    opts = qemu_opts_create(qemu_find_opts("chardev"), label, 1, &local_err);

    if (local_err) {

        error_report_err(local_err);

        return NULL;

    }



    if (strstart(filename, "mon:", &p)) {

        filename = p;

        qemu_opt_set(opts, "mux", "on", &error_abort);

        if (strcmp(filename, "stdio") == 0) {

            /* Monitor is muxed to stdio: do not exit on Ctrl+C by default

             * but pass it to the guest.  Handle this only for compat syntax,

             * for -chardev syntax we have special option for this.

             * This is what -nographic did, redirecting+muxing serial+monitor

             * to stdio causing Ctrl+C to be passed to guest. */

            qemu_opt_set(opts, "signal", "off", &error_abort);

        }

    }



    if (strcmp(filename, "null")    == 0 ||

        strcmp(filename, "pty")     == 0 ||

        strcmp(filename, "msmouse") == 0 ||

        strcmp(filename, "braille") == 0 ||

        strcmp(filename, "testdev") == 0 ||

        strcmp(filename, "stdio")   == 0) {

        qemu_opt_set(opts, "backend", filename, &error_abort);

        return opts;

    }

    if (strstart(filename, "vc", &p)) {

        qemu_opt_set(opts, "backend", "vc", &error_abort);

        if (*p == ':') {

            if (sscanf(p+1, "%7[0-9]x%7[0-9]", width, height) == 2) {

                /* pixels */

                qemu_opt_set(opts, "width", width, &error_abort);

                qemu_opt_set(opts, "height", height, &error_abort);

            } else if (sscanf(p+1, "%7[0-9]Cx%7[0-9]C", width, height) == 2) {

                /* chars */

                qemu_opt_set(opts, "cols", width, &error_abort);

                qemu_opt_set(opts, "rows", height, &error_abort);

            } else {

                goto fail;

            }

        }

        return opts;

    }

    if (strcmp(filename, "con:") == 0) {

        qemu_opt_set(opts, "backend", "console", &error_abort);

        return opts;

    }

    if (strstart(filename, "COM", NULL)) {

        qemu_opt_set(opts, "backend", "serial", &error_abort);

        qemu_opt_set(opts, "path", filename, &error_abort);

        return opts;

    }

    if (strstart(filename, "file:", &p)) {

        qemu_opt_set(opts, "backend", "file", &error_abort);

        qemu_opt_set(opts, "path", p, &error_abort);

        return opts;

    }

    if (strstart(filename, "pipe:", &p)) {

        qemu_opt_set(opts, "backend", "pipe", &error_abort);

        qemu_opt_set(opts, "path", p, &error_abort);

        return opts;

    }

    if (strstart(filename, "tcp:", &p) ||

        strstart(filename, "telnet:", &p)) {

        if (sscanf(p, "%64[^:]:%32[^,]%n", host, port, &pos) < 2) {

            host[0] = 0;

            if (sscanf(p, ":%32[^,]%n", port, &pos) < 1)

                goto fail;

        }

        qemu_opt_set(opts, "backend", "socket", &error_abort);

        qemu_opt_set(opts, "host", host, &error_abort);

        qemu_opt_set(opts, "port", port, &error_abort);

        if (p[pos] == ',') {

            if (qemu_opts_do_parse(opts, p+pos+1, NULL) != 0)

                goto fail;

        }

        if (strstart(filename, "telnet:", &p))

            qemu_opt_set(opts, "telnet", "on", &error_abort);

        return opts;

    }

    if (strstart(filename, "udp:", &p)) {

        qemu_opt_set(opts, "backend", "udp", &error_abort);

        if (sscanf(p, "%64[^:]:%32[^@,]%n", host, port, &pos) < 2) {

            host[0] = 0;

            if (sscanf(p, ":%32[^@,]%n", port, &pos) < 1) {

                goto fail;

            }

        }

        qemu_opt_set(opts, "host", host, &error_abort);

        qemu_opt_set(opts, "port", port, &error_abort);

        if (p[pos] == '@') {

            p += pos + 1;

            if (sscanf(p, "%64[^:]:%32[^,]%n", host, port, &pos) < 2) {

                host[0] = 0;

                if (sscanf(p, ":%32[^,]%n", port, &pos) < 1) {

                    goto fail;

                }

            }

            qemu_opt_set(opts, "localaddr", host, &error_abort);

            qemu_opt_set(opts, "localport", port, &error_abort);

        }

        return opts;

    }

    if (strstart(filename, "unix:", &p)) {

        qemu_opt_set(opts, "backend", "socket", &error_abort);

        if (qemu_opts_do_parse(opts, p, "path") != 0)

            goto fail;

        return opts;

    }

    if (strstart(filename, "/dev/parport", NULL) ||

        strstart(filename, "/dev/ppi", NULL)) {

        qemu_opt_set(opts, "backend", "parport", &error_abort);

        qemu_opt_set(opts, "path", filename, &error_abort);

        return opts;

    }

    if (strstart(filename, "/dev/", NULL)) {

        qemu_opt_set(opts, "backend", "tty", &error_abort);

        qemu_opt_set(opts, "path", filename, &error_abort);

        return opts;

    }



fail:

    qemu_opts_del(opts);

    return NULL;

}
