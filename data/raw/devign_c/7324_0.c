void ip6_input(struct mbuf *m)

{

    struct ip6 *ip6;



    DEBUG_CALL("ip6_input");

    DEBUG_ARG("m = %lx", (long)m);

    DEBUG_ARG("m_len = %d", m->m_len);



    if (m->m_len < sizeof(struct ip6)) {

        goto bad;

    }



    ip6 = mtod(m, struct ip6 *);



    if (ip6->ip_v != IP6VERSION) {

        goto bad;

    }



    /* check ip_ttl for a correct ICMP reply */

    if (ip6->ip_hl == 0) {

        /*icmp_send_error(m, ICMP_TIMXCEED,ICMP_TIMXCEED_INTRANS, 0,"ttl");*/

        goto bad;

    }



    /*

     * Switch out to protocol's input routine.

     */

    switch (ip6->ip_nh) {

    case IPPROTO_TCP:

        /*tcp_input(m, hlen, (struct socket *)NULL);*/

        break;

    case IPPROTO_UDP:

        /*udp_input(m, hlen);*/

        break;

    case IPPROTO_ICMPV6:

        icmp6_input(m);

        break;

    default:

        m_free(m);

    }

    return;

bad:

    m_free(m);

}
