static int net_socket_mcast_init(NetClientState *peer,

                                 const char *model,

                                 const char *name,

                                 const char *host_str,

                                 const char *localaddr_str)

{

    NetSocketState *s;

    int fd;

    struct sockaddr_in saddr;

    struct in_addr localaddr, *param_localaddr;



    if (parse_host_port(&saddr, host_str) < 0)

        return -1;



    if (localaddr_str != NULL) {

        if (inet_aton(localaddr_str, &localaddr) == 0)

            return -1;

        param_localaddr = &localaddr;

    } else {

        param_localaddr = NULL;

    }



    fd = net_socket_mcast_create(&saddr, param_localaddr);

    if (fd < 0)

        return -1;



    s = net_socket_fd_init(peer, model, name, fd, 0);

    if (!s)

        return -1;



    s->dgram_dst = saddr;



    snprintf(s->nc.info_str, sizeof(s->nc.info_str),

             "socket: mcast=%s:%d",

             inet_ntoa(saddr.sin_addr), ntohs(saddr.sin_port));

    return 0;



}
