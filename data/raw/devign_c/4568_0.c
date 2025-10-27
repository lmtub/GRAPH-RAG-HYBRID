int if_encap(Slirp *slirp, struct mbuf *ifm)

{

    uint8_t buf[1600];

    struct ethhdr *eh = (struct ethhdr *)buf;

    uint8_t ethaddr[ETH_ALEN];

    const struct ip *iph = (const struct ip *)ifm->m_data;

    int ret;



    if (ifm->m_len + ETH_HLEN > sizeof(buf)) {

        return 1;

    }



    switch (iph->ip_v) {

    case IPVERSION:

        ret = if_encap4(slirp, ifm, eh, ethaddr);

        if (ret < 2) {

            return ret;

        }

        break;



    default:

        /* Do not assert while we don't manage IP6VERSION */

        /* assert(0); */

        break;

    }



    memcpy(eh->h_dest, ethaddr, ETH_ALEN);

    DEBUG_ARGS((dfd, " src = %02x:%02x:%02x:%02x:%02x:%02x\n",

                eh->h_source[0], eh->h_source[1], eh->h_source[2],

                eh->h_source[3], eh->h_source[4], eh->h_source[5]));

    DEBUG_ARGS((dfd, " dst = %02x:%02x:%02x:%02x:%02x:%02x\n",

                eh->h_dest[0], eh->h_dest[1], eh->h_dest[2],

                eh->h_dest[3], eh->h_dest[4], eh->h_dest[5]));

    memcpy(buf + sizeof(struct ethhdr), ifm->m_data, ifm->m_len);

    slirp_output(slirp->opaque, buf, ifm->m_len + ETH_HLEN);

    return 1;

}
