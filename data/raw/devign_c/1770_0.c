static uint8_t net_tx_pkt_get_gso_type(struct NetTxPkt *pkt,

                                          bool tso_enable)

{

    uint8_t rc = VIRTIO_NET_HDR_GSO_NONE;

    uint16_t l3_proto;



    l3_proto = eth_get_l3_proto(pkt->vec[NET_TX_PKT_L2HDR_FRAG].iov_base,

        pkt->vec[NET_TX_PKT_L2HDR_FRAG].iov_len);



    if (!tso_enable) {

        goto func_exit;

    }



    rc = eth_get_gso_type(l3_proto, pkt->vec[NET_TX_PKT_L3HDR_FRAG].iov_base,

                          pkt->l4proto);



func_exit:

    return rc;

}
