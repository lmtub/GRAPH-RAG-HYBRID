void net_rx_pkt_attach_data(struct NetRxPkt *pkt, const void *data,

                               size_t len, bool strip_vlan)

{

    uint16_t tci = 0;

    uint16_t ploff;

    assert(pkt);

    pkt->vlan_stripped = false;



    if (strip_vlan) {

        pkt->vlan_stripped = eth_strip_vlan(data, pkt->ehdr_buf, &ploff, &tci);

    }



    if (pkt->vlan_stripped) {

        pkt->vec[0].iov_base = pkt->ehdr_buf;

        pkt->vec[0].iov_len = ploff - sizeof(struct vlan_header);

        pkt->vec[1].iov_base = (uint8_t *) data + ploff;

        pkt->vec[1].iov_len = len - ploff;

        pkt->vec_len = 2;

        pkt->tot_len = len - ploff + sizeof(struct eth_header);

    } else {

        pkt->vec[0].iov_base = (void *)data;

        pkt->vec[0].iov_len = len;

        pkt->vec_len = 1;

        pkt->tot_len = len;

    }



    pkt->tci = tci;

}
