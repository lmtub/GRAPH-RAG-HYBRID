void slirp_select_fill(int *pnfds,

                       fd_set *readfds, fd_set *writefds, fd_set *xfds)

{

    Slirp *slirp;

    struct socket *so, *so_next;

    int nfds;



    if (QTAILQ_EMPTY(&slirp_instances)) {

        return;

    }



    /* fail safe */

    global_readfds = NULL;

    global_writefds = NULL;

    global_xfds = NULL;



    nfds = *pnfds;

    /*

     * First, TCP sockets

     */

    do_slowtimo = 0;



    QTAILQ_FOREACH(slirp, &slirp_instances, entry) {

        /*

         * *_slowtimo needs calling if there are IP fragments

         * in the fragment queue, or there are TCP connections active

         */

        do_slowtimo |= ((slirp->tcb.so_next != &slirp->tcb) ||

                (&slirp->ipq.ip_link != slirp->ipq.ip_link.next));



        for (so = slirp->tcb.so_next; so != &slirp->tcb;

                so = so_next) {

            so_next = so->so_next;



            /*

             * See if we need a tcp_fasttimo

             */

            if (time_fasttimo == 0 && so->so_tcpcb->t_flags & TF_DELACK) {

                time_fasttimo = curtime; /* Flag when we want a fasttimo */

            }



            /*

             * NOFDREF can include still connecting to local-host,

             * newly socreated() sockets etc. Don't want to select these.

             */

            if (so->so_state & SS_NOFDREF || so->s == -1) {

                continue;

            }



            /*

             * Set for reading sockets which are accepting

             */

            if (so->so_state & SS_FACCEPTCONN) {

                FD_SET(so->s, readfds);

                UPD_NFDS(so->s);

                continue;

            }



            /*

             * Set for writing sockets which are connecting

             */

            if (so->so_state & SS_ISFCONNECTING) {

                FD_SET(so->s, writefds);

                UPD_NFDS(so->s);

                continue;

            }



            /*

             * Set for writing if we are connected, can send more, and

             * we have something to send

             */

            if (CONN_CANFSEND(so) && so->so_rcv.sb_cc) {

                FD_SET(so->s, writefds);

                UPD_NFDS(so->s);

            }



            /*

             * Set for reading (and urgent data) if we are connected, can

             * receive more, and we have room for it XXX /2 ?

             */

            if (CONN_CANFRCV(so) &&

                (so->so_snd.sb_cc < (so->so_snd.sb_datalen/2))) {

                FD_SET(so->s, readfds);

                FD_SET(so->s, xfds);

                UPD_NFDS(so->s);

            }

        }



        /*

         * UDP sockets

         */

        for (so = slirp->udb.so_next; so != &slirp->udb;

                so = so_next) {

            so_next = so->so_next;



            /*

             * See if it's timed out

             */

            if (so->so_expire) {

                if (so->so_expire <= curtime) {

                    udp_detach(so);

                    continue;

                } else {

                    do_slowtimo = 1; /* Let socket expire */

                }

            }



            /*

             * When UDP packets are received from over the

             * link, they're sendto()'d straight away, so

             * no need for setting for writing

             * Limit the number of packets queued by this session

             * to 4.  Note that even though we try and limit this

             * to 4 packets, the session could have more queued

             * if the packets needed to be fragmented

             * (XXX <= 4 ?)

             */

            if ((so->so_state & SS_ISFCONNECTED) && so->so_queued <= 4) {

                FD_SET(so->s, readfds);

                UPD_NFDS(so->s);

            }

        }



        /*

         * ICMP sockets

         */

        for (so = slirp->icmp.so_next; so != &slirp->icmp;

                so = so_next) {

            so_next = so->so_next;



            /*

             * See if it's timed out

             */

            if (so->so_expire) {

                if (so->so_expire <= curtime) {

                    icmp_detach(so);

                    continue;

                } else {

                    do_slowtimo = 1; /* Let socket expire */

                }

            }



            if (so->so_state & SS_ISFCONNECTED) {

                FD_SET(so->s, readfds);

                UPD_NFDS(so->s);

            }

        }

    }



    *pnfds = nfds;

}
