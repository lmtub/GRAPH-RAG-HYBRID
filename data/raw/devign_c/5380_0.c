static void net_tx_pkt_do_sw_csum(struct NetTxPkt *pkt)

{

    struct iovec *iov = &pkt->vec[NET_TX_PKT_L2HDR_FRAG];

    uint32_t csum_cntr;

    uint16_t csum = 0;

    /* num of iovec without vhdr */

    uint32_t iov_len = pkt->payload_frags + NET_TX_PKT_PL_START_FRAG - 1;

    uint16_t csl;

    struct ip_header *iphdr;

    size_t csum_offset = pkt->virt_hdr.csum_start + pkt->virt_hdr.csum_offset;



    /* Put zero to checksum field */

    iov_from_buf(iov, iov_len, csum_offset, &csum, sizeof csum);



    /* Calculate L4 TCP/UDP checksum */

    csl = pkt->payload_len;



    /* data checksum */

    csum_cntr =

        net_checksum_add_iov(iov, iov_len, pkt->virt_hdr.csum_start, csl);

    /* add pseudo header to csum */

    iphdr = pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_base;

    csum_cntr += eth_calc_pseudo_hdr_csum(iphdr, csl);



    /* Put the checksum obtained into the packet */

    csum = cpu_to_be16(net_checksum_finish(csum_cntr));

    iov_from_buf(iov, iov_len, csum_offset, &csum, sizeof csum);

}
