int unix_listen_opts(QemuOpts *opts)

{

    struct sockaddr_un un;

    const char *path = qemu_opt_get(opts, "path");

    int sock, fd;



    sock = socket(PF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {

        perror("socket(unix)");

        return -1;

    }



    memset(&un, 0, sizeof(un));

    un.sun_family = AF_UNIX;

    if (path && strlen(path)) {

        snprintf(un.sun_path, sizeof(un.sun_path), "%s", path);

    } else {

        char *tmpdir = getenv("TMPDIR");

        snprintf(un.sun_path, sizeof(un.sun_path), "%s/qemu-socket-XXXXXX",

                 tmpdir ? tmpdir : "/tmp");

        /*

         * This dummy fd usage silences the mktemp() unsecure warning.

         * Using mkstemp() doesn't make things more secure here

         * though.  bind() complains about existing files, so we have

         * to unlink first and thus re-open the race window.  The

         * worst case possible is bind() failing, i.e. a DoS attack.

         */

        fd = mkstemp(un.sun_path); close(fd);

        qemu_opt_set(opts, "path", un.sun_path);

    }



    unlink(un.sun_path);

    if (bind(sock, (struct sockaddr*) &un, sizeof(un)) < 0) {

        fprintf(stderr, "bind(unix:%s): %s\n", un.sun_path, strerror(errno));

        goto err;

    }

    if (listen(sock, 1) < 0) {

        fprintf(stderr, "listen(unix:%s): %s\n", un.sun_path, strerror(errno));

        goto err;

    }



    if (sockets_debug)

        fprintf(stderr, "bind(unix:%s): OK\n", un.sun_path);

    return sock;



err:

    closesocket(sock);

    return -1;

}
