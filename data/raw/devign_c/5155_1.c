int inet_listen_opts(QemuOpts *opts, int port_offset)

{

    struct addrinfo ai,*res,*e;

    const char *addr;

    char port[33];

    char uaddr[INET6_ADDRSTRLEN+1];

    char uport[33];

    int slisten,rc,to,try_next;



    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_PASSIVE | AI_ADDRCONFIG;

    ai.ai_family = PF_UNSPEC;

    ai.ai_socktype = SOCK_STREAM;



    if (qemu_opt_get(opts, "port") == NULL) {

        fprintf(stderr, "%s: host and/or port not specified\n", __FUNCTION__);

        return -1;

    }

    pstrcpy(port, sizeof(port), qemu_opt_get(opts, "port"));

    addr = qemu_opt_get(opts, "host");



    to = qemu_opt_get_number(opts, "to", 0);

    if (qemu_opt_get_bool(opts, "ipv4", 0))

        ai.ai_family = PF_INET;

    if (qemu_opt_get_bool(opts, "ipv6", 0))

        ai.ai_family = PF_INET6;



    /* lookup */

    if (port_offset)

        snprintf(port, sizeof(port), "%d", atoi(port) + port_offset);

    rc = getaddrinfo(strlen(addr) ? addr : NULL, port, &ai, &res);

    if (rc != 0) {

        fprintf(stderr,"getaddrinfo(%s,%s): %s\n", addr, port,

                gai_strerror(rc));

        return -1;

    }

    if (sockets_debug)

        inet_print_addrinfo(__FUNCTION__, res);



    /* create socket + bind */

    for (e = res; e != NULL; e = e->ai_next) {

        getnameinfo((struct sockaddr*)e->ai_addr,e->ai_addrlen,

		        uaddr,INET6_ADDRSTRLEN,uport,32,

		        NI_NUMERICHOST | NI_NUMERICSERV);

        slisten = socket(e->ai_family, e->ai_socktype, e->ai_protocol);

        if (slisten < 0) {

            fprintf(stderr,"%s: socket(%s): %s\n", __FUNCTION__,

                    inet_strfamily(e->ai_family), strerror(errno));

            continue;

        }



        setsockopt(slisten,SOL_SOCKET,SO_REUSEADDR,(void*)&on,sizeof(on));

#ifdef IPV6_V6ONLY

        if (e->ai_family == PF_INET6) {

            /* listen on both ipv4 and ipv6 */

            setsockopt(slisten,IPPROTO_IPV6,IPV6_V6ONLY,(void*)&off,

                sizeof(off));

        }

#endif



        for (;;) {

            if (bind(slisten, e->ai_addr, e->ai_addrlen) == 0) {

                if (sockets_debug)

                    fprintf(stderr,"%s: bind(%s,%s,%d): OK\n", __FUNCTION__,

                        inet_strfamily(e->ai_family), uaddr, inet_getport(e));

                goto listen;

            }

            try_next = to && (inet_getport(e) <= to + port_offset);

            if (!try_next || sockets_debug)

                fprintf(stderr,"%s: bind(%s,%s,%d): %s\n", __FUNCTION__,

                        inet_strfamily(e->ai_family), uaddr, inet_getport(e),

                        strerror(errno));

            if (try_next) {

                inet_setport(e, inet_getport(e) + 1);

                continue;

            }

            break;

        }

        closesocket(slisten);

    }

    fprintf(stderr, "%s: FAILED\n", __FUNCTION__);

    freeaddrinfo(res);

    return -1;



listen:

    if (listen(slisten,1) != 0) {

        perror("listen");

        closesocket(slisten);

        freeaddrinfo(res);

        return -1;

    }

    snprintf(uport, sizeof(uport), "%d", inet_getport(e) - port_offset);

    qemu_opt_set(opts, "host", uaddr);

    qemu_opt_set(opts, "port", uport);

    qemu_opt_set(opts, "ipv6", (e->ai_family == PF_INET6) ? "on" : "off");

    qemu_opt_set(opts, "ipv4", (e->ai_family != PF_INET6) ? "on" : "off");

    freeaddrinfo(res);

    return slisten;

}
