static int proxy_socket(const char *path, uid_t uid, gid_t gid)

{

    int sock, client;

    struct sockaddr_un proxy, qemu;

    socklen_t size;



    /* requested socket already exists, refuse to start */

    if (!access(path, F_OK)) {

        do_log(LOG_CRIT, "socket already exists\n");

        return -1;

    }




    sock = socket(AF_UNIX, SOCK_STREAM, 0);

    if (sock < 0) {

        do_perror("socket");

        return -1;

    }



    /* mask other part of mode bits */

    umask(7);



    proxy.sun_family = AF_UNIX;

    strcpy(proxy.sun_path, path);

    if (bind(sock, (struct sockaddr *)&proxy,

            sizeof(struct sockaddr_un)) < 0) {

        do_perror("bind");

        goto error;

    }

    if (chown(proxy.sun_path, uid, gid) < 0) {

        do_perror("chown");

        goto error;

    }

    if (listen(sock, 1) < 0) {

        do_perror("listen");

        goto error;

    }



    size = sizeof(qemu);

    client = accept(sock, (struct sockaddr *)&qemu, &size);

    if (client < 0) {

        do_perror("accept");

        goto error;

    }

    close(sock);

    return client;



error:

    close(sock);

    return -1;

}