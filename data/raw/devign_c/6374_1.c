static struct addrinfo *inet_parse_connect_saddr(InetSocketAddress *saddr,

                                                 Error **errp)

{

    struct addrinfo ai, *res;

    int rc;

    Error *err = NULL;



    memset(&ai, 0, sizeof(ai));



    ai.ai_flags = AI_CANONNAME | AI_V4MAPPED | AI_ADDRCONFIG;

    ai.ai_family = inet_ai_family_from_address(saddr, &err);

    ai.ai_socktype = SOCK_STREAM;



    if (err) {

        error_propagate(errp, err);

        return NULL;

    }



    if (saddr->host == NULL || saddr->port == NULL) {

        error_setg(errp, "host and/or port not specified");

        return NULL;

    }



    /* lookup */

    rc = getaddrinfo(saddr->host, saddr->port, &ai, &res);

    if (rc != 0) {

        error_setg(errp, "address resolution failed for %s:%s: %s",

                   saddr->host, saddr->port, gai_strerror(rc));

        return NULL;

    }

    return res;

}
