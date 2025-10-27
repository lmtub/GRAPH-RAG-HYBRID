int inet_dgram_opts(QemuOpts *opts, Error **errp)

{

    struct addrinfo ai, *peer = NULL, *local = NULL;

    const char *addr;

    const char *port;

    int sock = -1, rc;



    /* lookup peer addr */

    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_CANONNAME | AI_ADDRCONFIG;

    ai.ai_family = PF_UNSPEC;

    ai.ai_socktype = SOCK_DGRAM;



    addr = qemu_opt_get(opts, "host");

    port = qemu_opt_get(opts, "port");

    if (addr == NULL || strlen(addr) == 0) {

        addr = "localhost";

    }

    if (port == NULL || strlen(port) == 0) {

        error_setg(errp, "remote port not specified");

        return -1;

    }



    if (qemu_opt_get_bool(opts, "ipv4", 0))

        ai.ai_family = PF_INET;

    if (qemu_opt_get_bool(opts, "ipv6", 0))

        ai.ai_family = PF_INET6;



    if (0 != (rc = getaddrinfo(addr, port, &ai, &peer))) {

        error_setg(errp, "address resolution failed for %s:%s: %s", addr, port,

                   gai_strerror(rc));

	return -1;

    }



    /* lookup local addr */

    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_PASSIVE;

    ai.ai_family = peer->ai_family;

    ai.ai_socktype = SOCK_DGRAM;



    addr = qemu_opt_get(opts, "localaddr");

    port = qemu_opt_get(opts, "localport");

    if (addr == NULL || strlen(addr) == 0) {

        addr = NULL;

    }

    if (!port || strlen(port) == 0)

        port = "0";



    if (0 != (rc = getaddrinfo(addr, port, &ai, &local))) {

        error_setg(errp, "address resolution failed for %s:%s: %s", addr, port,

                   gai_strerror(rc));

        goto err;

    }



    /* create socket */

    sock = qemu_socket(peer->ai_family, peer->ai_socktype, peer->ai_protocol);

    if (sock < 0) {

        error_setg_errno(errp, errno, "Failed to create socket");

        goto err;

    }

    socket_set_fast_reuse(sock);



    /* bind socket */

    if (bind(sock, local->ai_addr, local->ai_addrlen) < 0) {

        error_setg_errno(errp, errno, "Failed to bind socket");

        goto err;

    }



    /* connect to peer */

    if (connect(sock,peer->ai_addr,peer->ai_addrlen) < 0) {

        error_setg_errno(errp, errno, "Failed to connect socket");

        goto err;

    }



    freeaddrinfo(local);

    freeaddrinfo(peer);

    return sock;



err:

    if (-1 != sock)

        closesocket(sock);

    if (local)

        freeaddrinfo(local);

    if (peer)

        freeaddrinfo(peer);

    return -1;

}
