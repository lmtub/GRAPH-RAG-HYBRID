int inet_connect_opts(QemuOpts *opts, bool block, bool *in_progress,

                      Error **errp)

{

    struct addrinfo *res, *e;

    int sock = -1;



    res = inet_parse_connect_opts(opts, errp);

    if (!res) {

        return -1;

    }



    if (in_progress) {

        *in_progress = false;

    }



    for (e = res; e != NULL; e = e->ai_next) {

        sock = inet_connect_addr(e, block, in_progress);

        if (sock >= 0) {

            break;

        }

    }

    if (sock < 0) {

        error_set(errp, QERR_SOCKET_CONNECT_FAILED);

    }

    freeaddrinfo(res);

    return sock;

}
