static NetSocketState *net_socket_fd_init_dgram(NetClientState *peer,

                                                const char *model,

                                                const char *name,

                                                int fd, int is_connected,

                                                const char *mcast,

                                                Error **errp)

{

    struct sockaddr_in saddr;

    int newfd;

    NetClientState *nc;

    NetSocketState *s;



    /* fd passed: multicast: "learn" dgram_dst address from bound address and save it

     * Because this may be "shared" socket from a "master" process, datagrams would be recv()

     * by ONLY ONE process: we must "clone" this dgram socket --jjo

     */



    if (is_connected && mcast != NULL) {

            if (parse_host_port(&saddr, mcast, errp) < 0) {

                goto err;

            }

            /* must be bound */

            if (saddr.sin_addr.s_addr == 0) {

                error_setg(errp, "can't setup multicast destination address");

                goto err;

            }

            /* clone dgram socket */

            newfd = net_socket_mcast_create(&saddr, NULL, errp);

            if (newfd < 0) {

                goto err;

            }

            /* clone newfd to fd, close newfd */

            dup2(newfd, fd);

            close(newfd);



    }



    nc = qemu_new_net_client(&net_dgram_socket_info, peer, model, name);



    s = DO_UPCAST(NetSocketState, nc, nc);



    s->fd = fd;

    s->listen_fd = -1;

    s->send_fn = net_socket_send_dgram;

    net_socket_rs_init(&s->rs, net_socket_rs_finalize, false);

    net_socket_read_poll(s, true);



    /* mcast: save bound address as dst */

    if (is_connected) {

        s->dgram_dst = saddr;

        snprintf(nc->info_str, sizeof(nc->info_str),

                 "socket: fd=%d (cloned mcast=%s:%d)",

                 fd, inet_ntoa(saddr.sin_addr), ntohs(saddr.sin_port));

    } else {

        snprintf(nc->info_str, sizeof(nc->info_str),

                 "socket: fd=%d", fd);

    }



    return s;



err:

    closesocket(fd);

    return NULL;

}
