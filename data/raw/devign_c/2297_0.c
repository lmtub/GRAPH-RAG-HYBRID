bool net_tx_pkt_add_raw_fragment(struct NetTxPkt *pkt, hwaddr pa,

    size_t len)

{

    hwaddr mapped_len = 0;

    struct iovec *ventry;

    assert(pkt);

    assert(pkt->max_raw_frags > pkt->raw_frags);



    if (!len) {

        return true;

     }



    ventry = &pkt->raw[pkt->raw_frags];

    mapped_len = len;



    ventry->iov_base = cpu_physical_memory_map(pa, &mapped_len, false);

    ventry->iov_len = mapped_len;

    pkt->raw_frags += !!ventry->iov_base;



    if ((ventry->iov_base == NULL) || (len != mapped_len)) {

        return false;

    }



    return true;

}
