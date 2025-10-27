static int check_protocol_support(bool *has_ipv4, bool *has_ipv6)

{

    struct sockaddr_in sin = {

        .sin_family = AF_INET,

        .sin_addr = { .s_addr = htonl(INADDR_LOOPBACK) },

    };

    struct sockaddr_in6 sin6 = {

        .sin6_family = AF_INET6,

        .sin6_addr = IN6ADDR_LOOPBACK_INIT,

    };



    if (check_bind((struct sockaddr *)&sin, sizeof(sin), has_ipv4) < 0) {

        return -1;

    }

    if (check_bind((struct sockaddr *)&sin6, sizeof(sin6), has_ipv6) < 0) {

        return -1;

    }



    return 0;

}
