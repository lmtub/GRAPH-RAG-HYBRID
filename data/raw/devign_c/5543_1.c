e1000e_write_ps_rx_descr(E1000ECore *core, uint8_t *desc,

                         struct NetRxPkt *pkt,

                         const E1000E_RSSInfo *rss_info,

                         size_t ps_hdr_len,

                         uint16_t(*written)[MAX_PS_BUFFERS])

{

    int i;

    union e1000_rx_desc_packet_split *d =

        (union e1000_rx_desc_packet_split *) desc;



    memset(d, 0, sizeof(*d));



    d->wb.middle.length0 = cpu_to_le16((*written)[0]);



    for (i = 0; i < PS_PAGE_BUFFERS; i++) {

        d->wb.upper.length[i] = cpu_to_le16((*written)[i + 1]);

    }



    e1000e_build_rx_metadata(core, pkt, pkt != NULL,

                             rss_info,

                             &d->wb.lower.hi_dword.rss,

                             &d->wb.lower.mrq,

                             &d->wb.middle.status_error,

                             &d->wb.lower.hi_dword.csum_ip.ip_id,

                             &d->wb.middle.vlan);



    d->wb.upper.header_status =

        cpu_to_le16(ps_hdr_len | (ps_hdr_len ? E1000_RXDPS_HDRSTAT_HDRSP : 0));



    trace_e1000e_rx_desc_ps_write((*written)[0], (*written)[1],

                                  (*written)[2], (*written)[3]);

}
