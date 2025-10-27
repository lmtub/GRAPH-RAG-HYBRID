static int handle_secondary_tcp_pkt(NetFilterState *nf,

                                    Connection *conn,

                                    Packet *pkt)

{

    struct tcphdr *tcp_pkt;



    tcp_pkt = (struct tcphdr *)pkt->transport_header;



    if (trace_event_get_state(TRACE_COLO_FILTER_REWRITER_DEBUG)) {

        trace_colo_filter_rewriter_pkt_info(__func__,

                    inet_ntoa(pkt->ip->ip_src), inet_ntoa(pkt->ip->ip_dst),

                    ntohl(tcp_pkt->th_seq), ntohl(tcp_pkt->th_ack),

                    tcp_pkt->th_flags);

        trace_colo_filter_rewriter_conn_offset(conn->offset);

    }



    if (((tcp_pkt->th_flags & (TH_ACK | TH_SYN)) == (TH_ACK | TH_SYN))) {

        /*

         * save offset = secondary_seq and then

         * in handle_primary_tcp_pkt make offset

         * = secondary_seq - primary_seq

         */

        conn->offset = ntohl(tcp_pkt->th_seq);

    }



    if ((tcp_pkt->th_flags & (TH_ACK | TH_SYN)) == TH_ACK) {

        /* handle packets to the primary from the secondary*/

        tcp_pkt->th_seq = htonl(ntohl(tcp_pkt->th_seq) - conn->offset);



        net_checksum_calculate((uint8_t *)pkt->data, pkt->size);

    }



    return 0;

}
