int inet_dgram_opts(QemuOpts *opts)

{

    struct addrinfo ai, *peer = NULL, *local = NULL;

    const char *addr;

    const char *port;

    char uaddr[INET6_ADDRSTRLEN+1];

    char uport[33];

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

        fprintf(stderr, "inet_dgram: port not specified\n");

        return -1;

    }



    if (qemu_opt_get_bool(opts, "ipv4", 0))

        ai.ai_family = PF_INET;

    if (qemu_opt_get_bool(opts, "ipv6", 0))

        ai.ai_family = PF_INET6;



    if (0 != (rc = getaddrinfo(addr, port, &ai, &peer))) {

        fprintf(stderr,"getaddrinfo(%s,%s): %s\n", addr, port,

                gai_strerror(rc));

	return -1;

    }

    if (sockets_debug) {

        fprintf(stderr, "%s: peer (%s:%s)\n", __FUNCTION__, addr, port);

        inet_print_addrinfo(__FUNCTION__, peer);

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

        fprintf(stderr,"getaddrinfo(%s,%s): %s\n", addr, port,

                gai_strerror(rc));

        return -1;

    }

    if (sockets_debug) {

        fprintf(stderr, "%s: local (%s:%s)\n", __FUNCTION__, addr, port);

        inet_print_addrinfo(__FUNCTION__, local);

    }



    /* create socket */

    sock = socket(peer->ai_family, peer->ai_socktype, peer->ai_protocol);

    if (sock < 0) {

        fprintf(stderr,"%s: socket(%s): %s\n", __FUNCTION__,

                inet_strfamily(peer->ai_family), strerror(errno));

        goto err;

    }

    setsockopt(sock,SOL_SOCKET,SO_REUSEADDR,(void*)&on,sizeof(on));



    /* bind socket */

    if (getnameinfo((struct sockaddr*)local->ai_addr,local->ai_addrlen,

                    uaddr,INET6_ADDRSTRLEN,uport,32,

                    NI_NUMERICHOST | NI_NUMERICSERV) != 0) {

        fprintf(stderr, "%s: getnameinfo: oops\n", __FUNCTION__);

        goto err;

    }

    if (bind(sock, local->ai_addr, local->ai_addrlen) < 0) {

        fprintf(stderr,"%s: bind(%s,%s,%d): OK\n", __FUNCTION__,

                inet_strfamily(local->ai_family), uaddr, inet_getport(local));

        goto err;

    }



    /* connect to peer */

    if (getnameinfo((struct sockaddr*)peer->ai_addr, peer->ai_addrlen,

                    uaddr, INET6_ADDRSTRLEN, uport, 32,

                    NI_NUMERICHOST | NI_NUMERICSERV) != 0) {

        fprintf(stderr, "%s: getnameinfo: oops\n", __FUNCTION__);

        goto err;

    }

    if (connect(sock,peer->ai_addr,peer->ai_addrlen) < 0) {

        fprintf(stderr, "%s: connect(%s,%s,%s,%s): %s\n", __FUNCTION__,

                inet_strfamily(peer->ai_family),

                peer->ai_canonname, uaddr, uport, strerror(errno));

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
