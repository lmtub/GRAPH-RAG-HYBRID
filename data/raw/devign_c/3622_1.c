e1000e_write_lgcy_rx_descr(E1000ECore *core, uint8_t *desc,

                           struct NetRxPkt *pkt,

                           const E1000E_RSSInfo *rss_info,

                           uint16_t length)

{

    uint32_t status_flags, rss, mrq;

    uint16_t ip_id;



    struct e1000_rx_desc *d = (struct e1000_rx_desc *) desc;



    memset(d, 0, sizeof(*d));



    assert(!rss_info->enabled);



    d->length = cpu_to_le16(length);



    e1000e_build_rx_metadata(core, pkt, pkt != NULL,

                             rss_info,

                             &rss, &mrq,

                             &status_flags, &ip_id,

                             &d->special);

    d->errors = (uint8_t) (le32_to_cpu(status_flags) >> 24);

    d->status = (uint8_t) le32_to_cpu(status_flags);

}
