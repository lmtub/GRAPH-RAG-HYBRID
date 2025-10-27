static struct addrinfo *inet_parse_connect_opts(QemuOpts *opts, Error **errp)

{

    struct addrinfo ai, *res;

    int rc;

    const char *addr;

    const char *port;



    memset(&ai, 0, sizeof(ai));



    ai.ai_flags = AI_CANONNAME | AI_ADDRCONFIG;

    ai.ai_family = PF_UNSPEC;

    ai.ai_socktype = SOCK_STREAM;



    addr = qemu_opt_get(opts, "host");

    port = qemu_opt_get(opts, "port");

    if (addr == NULL || port == NULL) {

        error_setg(errp, "host and/or port not specified");

        return NULL;

    }



    if (qemu_opt_get_bool(opts, "ipv4", 0)) {

        ai.ai_family = PF_INET;

    }

    if (qemu_opt_get_bool(opts, "ipv6", 0)) {

        ai.ai_family = PF_INET6;

    }



    /* lookup */

    rc = getaddrinfo(addr, port, &ai, &res);

    if (rc != 0) {

        error_setg(errp, "address resolution failed for %s:%s: %s", addr, port,

                   gai_strerror(rc));

        return NULL;

    }

    return res;

}
