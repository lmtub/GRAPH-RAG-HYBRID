static void vmxnet3_rx_need_csum_calculate(struct VmxnetRxPkt *pkt,

                                           const void *pkt_data,

                                           size_t pkt_len)

{

    struct virtio_net_hdr *vhdr;

    bool isip4, isip6, istcp, isudp;

    uint8_t *data;

    int len;



    if (!vmxnet_rx_pkt_has_virt_hdr(pkt)) {

        return;

    }



    vhdr = vmxnet_rx_pkt_get_vhdr(pkt);

    if (!VMXNET_FLAG_IS_SET(vhdr->flags, VIRTIO_NET_HDR_F_NEEDS_CSUM)) {

        return;

    }



    vmxnet_rx_pkt_get_protocols(pkt, &isip4, &isip6, &isudp, &istcp);

    if (!(isip4 || isip6) || !(istcp || isudp)) {

        return;

    }



    vmxnet3_dump_virt_hdr(vhdr);



    /* Validate packet len: csum_start + scum_offset + length of csum field */

    if (pkt_len < (vhdr->csum_start + vhdr->csum_offset + 2)) {

        VMW_PKPRN("packet len:%d < csum_start(%d) + csum_offset(%d) + 2, "

                  "cannot calculate checksum",

                  len, vhdr->csum_start, vhdr->csum_offset);

        return;

    }



    data = (uint8_t *)pkt_data + vhdr->csum_start;

    len = pkt_len - vhdr->csum_start;

    /* Put the checksum obtained into the packet */

    stw_be_p(data + vhdr->csum_offset, net_raw_checksum(data, len));



    vhdr->flags &= ~VIRTIO_NET_HDR_F_NEEDS_CSUM;

    vhdr->flags |= VIRTIO_NET_HDR_F_DATA_VALID;

}
