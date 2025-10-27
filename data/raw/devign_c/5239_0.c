void tcp_connect(struct socket *inso)

{

    Slirp *slirp = inso->slirp;

    struct socket *so;

    struct sockaddr_in addr;

    socklen_t addrlen = sizeof(struct sockaddr_in);

    struct tcpcb *tp;

    int s, opt;



    DEBUG_CALL("tcp_connect");

    DEBUG_ARG("inso = %p", inso);



    /*

     * If it's an SS_ACCEPTONCE socket, no need to socreate()

     * another socket, just use the accept() socket.

     */

    if (inso->so_state & SS_FACCEPTONCE) {

        /* FACCEPTONCE already have a tcpcb */

        so = inso;

    } else {

        so = socreate(slirp);

        if (so == NULL) {

            /* If it failed, get rid of the pending connection */

            closesocket(accept(inso->s, (struct sockaddr *)&addr, &addrlen));

            return;

        }

        if (tcp_attach(so) < 0) {

            free(so); /* NOT sofree */

            return;

        }

        so->so_lfamily = AF_INET;

        so->so_laddr = inso->so_laddr;

        so->so_lport = inso->so_lport;

    }



    tcp_mss(sototcpcb(so), 0);



    s = accept(inso->s, (struct sockaddr *)&addr, &addrlen);

    if (s < 0) {

        tcp_close(sototcpcb(so)); /* This will sofree() as well */

        return;

    }

    qemu_set_nonblock(s);

    socket_set_fast_reuse(s);

    opt = 1;

    qemu_setsockopt(s, SOL_SOCKET, SO_OOBINLINE, &opt, sizeof(int));

    socket_set_nodelay(s);



    so->so_ffamily = AF_INET;

    so->so_fport = addr.sin_port;

    so->so_faddr = addr.sin_addr;

    /* Translate connections from localhost to the real hostname */

    if (so->so_faddr.s_addr == 0 ||

        (so->so_faddr.s_addr & loopback_mask) ==

        (loopback_addr.s_addr & loopback_mask)) {

        so->so_faddr = slirp->vhost_addr;

    }



    /* Close the accept() socket, set right state */

    if (inso->so_state & SS_FACCEPTONCE) {

        /* If we only accept once, close the accept() socket */

        closesocket(so->s);



        /* Don't select it yet, even though we have an FD */

        /* if it's not FACCEPTONCE, it's already NOFDREF */

        so->so_state = SS_NOFDREF;

    }

    so->s = s;

    so->so_state |= SS_INCOMING;



    so->so_iptos = tcp_tos(so);

    tp = sototcpcb(so);



    tcp_template(tp);



    tp->t_state = TCPS_SYN_SENT;

    tp->t_timer[TCPT_KEEP] = TCPTV_KEEP_INIT;

    tp->iss = slirp->tcp_iss;

    slirp->tcp_iss += TCP_ISSINCR/2;

    tcp_sendseqinit(tp);

    tcp_output(tp);

}
