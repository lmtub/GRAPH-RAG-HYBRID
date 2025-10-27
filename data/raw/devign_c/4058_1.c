_eth_get_rss_ex_dst_addr(const struct iovec *pkt, int pkt_frags,

                        size_t rthdr_offset,

                        struct ip6_ext_hdr *ext_hdr,

                        struct in6_address *dst_addr)

{

    struct ip6_ext_hdr_routing *rthdr = (struct ip6_ext_hdr_routing *) ext_hdr;



    if ((rthdr->rtype == 2) &&

        (rthdr->len == sizeof(struct in6_address) / 8) &&

        (rthdr->segleft == 1)) {



        size_t input_size = iov_size(pkt, pkt_frags);

        size_t bytes_read;



        if (input_size < rthdr_offset + sizeof(*ext_hdr)) {

            return false;

        }



        bytes_read = iov_to_buf(pkt, pkt_frags,

                                rthdr_offset + sizeof(*ext_hdr),

                                dst_addr, sizeof(*dst_addr));



        return bytes_read == sizeof(dst_addr);

    }



    return false;

}
