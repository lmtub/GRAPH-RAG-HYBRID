void net_tx_pkt_update_ip_checksums(struct NetTxPkt *pkt)

{

    uint16_t csum;

    uint32_t ph_raw_csum;

    assert(pkt);

    uint8_t gso_type = pkt->virt_hdr.gso_type & ~VIRTIO_NET_HDR_GSO_ECN;

    struct ip_header *ip_hdr;



    if (VIRTIO_NET_HDR_GSO_TCPV4 != gso_type &&

        VIRTIO_NET_HDR_GSO_UDP != gso_type) {

        return;

    }



    ip_hdr = pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_base;



    if (pkt->payload_len + pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_len >

        ETH_MAX_IP_DGRAM_LEN) {

        return;

    }



    ip_hdr->ip_len = cpu_to_be16(pkt->payload_len +

        pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_len);



    /* Calculate IP header checksum                    */

    ip_hdr->ip_sum = 0;

    csum = net_raw_checksum((uint8_t *)ip_hdr,

        pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_len);

    ip_hdr->ip_sum = cpu_to_be16(csum);



    /* Calculate IP pseudo header checksum             */

    ph_raw_csum = eth_calc_pseudo_hdr_csum(ip_hdr, pkt->payload_len);

    csum = cpu_to_be16(~net_checksum_finish(ph_raw_csum));

    iov_from_buf(&pkt->vec[NET_TX_PKT_PL_START_FRAG], pkt->payload_frags,

                 pkt->virt_hdr.csum_offset, &csum, sizeof(csum));

}
