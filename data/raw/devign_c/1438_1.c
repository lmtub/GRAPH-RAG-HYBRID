static int inet_listen_saddr(InetSocketAddress *saddr,

                             int port_offset,

                             bool update_addr,

                             Error **errp)

{

    struct addrinfo ai,*res,*e;

    char port[33];

    char uaddr[INET6_ADDRSTRLEN+1];

    char uport[33];

    int rc, port_min, port_max, p;

    int slisten = 0;

    int saved_errno = 0;

    bool socket_created = false;

    Error *err = NULL;



    memset(&ai,0, sizeof(ai));

    ai.ai_flags = AI_PASSIVE;

    if (saddr->has_numeric && saddr->numeric) {

        ai.ai_flags |= AI_NUMERICHOST | AI_NUMERICSERV;

    }

    ai.ai_family = inet_ai_family_from_address(saddr, &err);

    ai.ai_socktype = SOCK_STREAM;



    if (err) {

        error_propagate(errp, err);

        return -1;

    }



    if (saddr->host == NULL) {

        error_setg(errp, "host not specified");

        return -1;

    }

    if (saddr->port != NULL) {

        pstrcpy(port, sizeof(port), saddr->port);

    } else {

        port[0] = '\0';

    }



    /* lookup */

    if (port_offset) {

        unsigned long long baseport;

        if (strlen(port) == 0) {

            error_setg(errp, "port not specified");

            return -1;

        }

        if (parse_uint_full(port, &baseport, 10) < 0) {

            error_setg(errp, "can't convert to a number: %s", port);

            return -1;

        }

        if (baseport > 65535 ||

            baseport + port_offset > 65535) {

            error_setg(errp, "port %s out of range", port);

            return -1;

        }

        snprintf(port, sizeof(port), "%d", (int)baseport + port_offset);

    }

    rc = getaddrinfo(strlen(saddr->host) ? saddr->host : NULL,

                     strlen(port) ? port : NULL, &ai, &res);

    if (rc != 0) {

        error_setg(errp, "address resolution failed for %s:%s: %s",

                   saddr->host, port, gai_strerror(rc));

        return -1;

    }



    /* create socket + bind/listen */

    for (e = res; e != NULL; e = e->ai_next) {

        getnameinfo((struct sockaddr*)e->ai_addr,e->ai_addrlen,

		        uaddr,INET6_ADDRSTRLEN,uport,32,

		        NI_NUMERICHOST | NI_NUMERICSERV);



        slisten = create_fast_reuse_socket(e);

        if (slisten < 0) {

            continue;

        }



        socket_created = true;

        port_min = inet_getport(e);

        port_max = saddr->has_to ? saddr->to + port_offset : port_min;

        for (p = port_min; p <= port_max; p++) {

            inet_setport(e, p);

            rc = try_bind(slisten, saddr, e);

            if (rc) {

                if (errno == EADDRINUSE) {

                    continue;

                } else {

                    error_setg_errno(errp, errno, "Failed to bind socket");

                    goto listen_failed;

                }

            }

            if (!listen(slisten, 1)) {

                goto listen_ok;

            }

            if (errno != EADDRINUSE) {

                error_setg_errno(errp, errno, "Failed to listen on socket");

                goto listen_failed;

            }

            /* Someone else managed to bind to the same port and beat us

             * to listen on it! Socket semantics does not allow us to

             * recover from this situation, so we need to recreate the

             * socket to allow bind attempts for subsequent ports:

             */

            closesocket(slisten);

            slisten = create_fast_reuse_socket(e);

            if (slisten < 0) {

                error_setg_errno(errp, errno,

                                 "Failed to recreate failed listening socket");

                goto listen_failed;

            }

        }

    }

    error_setg_errno(errp, errno,

                     socket_created ?

                     "Failed to find an available port" :

                     "Failed to create a socket");

listen_failed:

    saved_errno = errno;

    if (slisten >= 0) {

        closesocket(slisten);

    }

    freeaddrinfo(res);

    errno = saved_errno;

    return -1;



listen_ok:

    if (update_addr) {

        g_free(saddr->host);

        saddr->host = g_strdup(uaddr);

        g_free(saddr->port);

        saddr->port = g_strdup_printf("%d",

                                      inet_getport(e) - port_offset);

        saddr->has_ipv6 = saddr->ipv6 = e->ai_family == PF_INET6;

        saddr->has_ipv4 = saddr->ipv4 = e->ai_family != PF_INET6;

    }

    freeaddrinfo(res);

    return slisten;

}
