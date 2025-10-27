e1000e_write_ext_rx_descr(E1000ECore *core, uint8_t *desc,

                          struct NetRxPkt *pkt,

                          const E1000E_RSSInfo *rss_info,

                          uint16_t length)

{

    union e1000_rx_desc_extended *d = (union e1000_rx_desc_extended *) desc;



    memset(d, 0, sizeof(*d));



    d->wb.upper.length = cpu_to_le16(length);



    e1000e_build_rx_metadata(core, pkt, pkt != NULL,

                             rss_info,

                             &d->wb.lower.hi_dword.rss,

                             &d->wb.lower.mrq,

                             &d->wb.upper.status_error,

                             &d->wb.lower.hi_dword.csum_ip.ip_id,

                             &d->wb.upper.vlan);

}
