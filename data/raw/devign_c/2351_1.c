void slirp_pollfds_poll(GArray *pollfds, int select_error)

{

    Slirp *slirp;

    struct socket *so, *so_next;

    int ret;



    if (QTAILQ_EMPTY(&slirp_instances)) {

        return;

    }



    curtime = qemu_clock_get_ms(QEMU_CLOCK_REALTIME);



    QTAILQ_FOREACH(slirp, &slirp_instances, entry) {

        /*

         * See if anything has timed out

         */

        if (slirp->time_fasttimo &&

            ((curtime - slirp->time_fasttimo) >= TIMEOUT_FAST)) {

            tcp_fasttimo(slirp);

            slirp->time_fasttimo = 0;

        }

        if (slirp->do_slowtimo &&

            ((curtime - slirp->last_slowtimo) >= TIMEOUT_SLOW)) {

            ip_slowtimo(slirp);

            tcp_slowtimo(slirp);

            slirp->last_slowtimo = curtime;

        }



        /*

         * Check sockets

         */

        if (!select_error) {

            /*

             * Check TCP sockets

             */

            for (so = slirp->tcb.so_next; so != &slirp->tcb;

                    so = so_next) {

                int revents;



                so_next = so->so_next;



                revents = 0;

                if (so->pollfds_idx != -1) {

                    revents = g_array_index(pollfds, GPollFD,

                                            so->pollfds_idx).revents;

                }



                if (so->so_state & SS_NOFDREF || so->s == -1) {

                    continue;

                }



                /*

                 * Check for URG data

                 * This will soread as well, so no need to

                 * test for G_IO_IN below if this succeeds

                 */

                if (revents & G_IO_PRI) {

                    sorecvoob(so);

                }

                /*

                 * Check sockets for reading

                 */

                else if (revents & (G_IO_IN | G_IO_HUP | G_IO_ERR)) {

                    /*

                     * Check for incoming connections

                     */

                    if (so->so_state & SS_FACCEPTCONN) {

                        tcp_connect(so);

                        continue;

                    } /* else */

                    ret = soread(so);



                    /* Output it if we read something */

                    if (ret > 0) {

                        tcp_output(sototcpcb(so));

                    }

                }



                /*

                 * Check sockets for writing

                 */

                if (!(so->so_state & SS_NOFDREF) &&

                        (revents & (G_IO_OUT | G_IO_ERR))) {

                    /*

                     * Check for non-blocking, still-connecting sockets

                     */

                    if (so->so_state & SS_ISFCONNECTING) {

                        /* Connected */

                        so->so_state &= ~SS_ISFCONNECTING;



                        ret = send(so->s, (const void *) &ret, 0, 0);

                        if (ret < 0) {

                            /* XXXXX Must fix, zero bytes is a NOP */

                            if (errno == EAGAIN || errno == EWOULDBLOCK ||

                                errno == EINPROGRESS || errno == ENOTCONN) {

                                continue;

                            }



                            /* else failed */

                            so->so_state &= SS_PERSISTENT_MASK;

                            so->so_state |= SS_NOFDREF;

                        }

                        /* else so->so_state &= ~SS_ISFCONNECTING; */



                        /*

                         * Continue tcp_input

                         */

                        tcp_input((struct mbuf *)NULL, sizeof(struct ip), so,

                                  so->so_ffamily);

                        /* continue; */

                    } else {

                        ret = sowrite(so);

                    }

                    /*

                     * XXXXX If we wrote something (a lot), there

                     * could be a need for a window update.

                     * In the worst case, the remote will send

                     * a window probe to get things going again

                     */

                }



                /*

                 * Probe a still-connecting, non-blocking socket

                 * to check if it's still alive

                 */

#ifdef PROBE_CONN

                if (so->so_state & SS_ISFCONNECTING) {

                    ret = qemu_recv(so->s, &ret, 0, 0);



                    if (ret < 0) {

                        /* XXX */

                        if (errno == EAGAIN || errno == EWOULDBLOCK ||

                            errno == EINPROGRESS || errno == ENOTCONN) {

                            continue; /* Still connecting, continue */

                        }



                        /* else failed */

                        so->so_state &= SS_PERSISTENT_MASK;

                        so->so_state |= SS_NOFDREF;



                        /* tcp_input will take care of it */

                    } else {

                        ret = send(so->s, &ret, 0, 0);

                        if (ret < 0) {

                            /* XXX */

                            if (errno == EAGAIN || errno == EWOULDBLOCK ||

                                errno == EINPROGRESS || errno == ENOTCONN) {

                                continue;

                            }

                            /* else failed */

                            so->so_state &= SS_PERSISTENT_MASK;

                            so->so_state |= SS_NOFDREF;

                        } else {

                            so->so_state &= ~SS_ISFCONNECTING;

                        }



                    }

                    tcp_input((struct mbuf *)NULL, sizeof(struct ip), so,

                              so->so_ffamily);

                } /* SS_ISFCONNECTING */

#endif

            }



            /*

             * Now UDP sockets.

             * Incoming packets are sent straight away, they're not buffered.

             * Incoming UDP data isn't buffered either.

             */

            for (so = slirp->udb.so_next; so != &slirp->udb;

                    so = so_next) {

                int revents;



                so_next = so->so_next;



                revents = 0;

                if (so->pollfds_idx != -1) {

                    revents = g_array_index(pollfds, GPollFD,

                            so->pollfds_idx).revents;

                }



                if (so->s != -1 &&

                    (revents & (G_IO_IN | G_IO_HUP | G_IO_ERR))) {

                    sorecvfrom(so);

                }

            }



            /*

             * Check incoming ICMP relies.

             */

            for (so = slirp->icmp.so_next; so != &slirp->icmp;

                    so = so_next) {

                    int revents;



                    so_next = so->so_next;



                    revents = 0;

                    if (so->pollfds_idx != -1) {

                        revents = g_array_index(pollfds, GPollFD,

                                                so->pollfds_idx).revents;

                    }



                    if (so->s != -1 &&

                        (revents & (G_IO_IN | G_IO_HUP | G_IO_ERR))) {

                    icmp_receive(so);

                }

            }

        }



        if_start(slirp);

    }

}
