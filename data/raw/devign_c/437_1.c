static int handle_primary_tcp_pkt(NetFilterState *nf,

                                  Connection *conn,

                                  Packet *pkt)

{

    struct tcphdr *tcp_pkt;



    tcp_pkt = (struct tcphdr *)pkt->transport_header;

    if (trace_event_get_state_backends(TRACE_COLO_FILTER_REWRITER_DEBUG)) {

        trace_colo_filter_rewriter_pkt_info(__func__,

                    inet_ntoa(pkt->ip->ip_src), inet_ntoa(pkt->ip->ip_dst),

                    ntohl(tcp_pkt->th_seq), ntohl(tcp_pkt->th_ack),

                    tcp_pkt->th_flags);

        trace_colo_filter_rewriter_conn_offset(conn->offset);

    }



    if (((tcp_pkt->th_flags & (TH_ACK | TH_SYN)) == TH_SYN)) {

        /*

         * we use this flag update offset func

         * run once in independent tcp connection

         */

        conn->syn_flag = 1;

    }



    if (((tcp_pkt->th_flags & (TH_ACK | TH_SYN)) == TH_ACK)) {

        if (conn->syn_flag) {

            /*

             * offset = secondary_seq - primary seq

             * ack packet sent by guest from primary node,

             * so we use th_ack - 1 get primary_seq

             */

            conn->offset -= (ntohl(tcp_pkt->th_ack) - 1);

            conn->syn_flag = 0;

        }

        if (conn->offset) {

            /* handle packets to the secondary from the primary */

            tcp_pkt->th_ack = htonl(ntohl(tcp_pkt->th_ack) + conn->offset);



            net_checksum_calculate((uint8_t *)pkt->data, pkt->size);

        }

    }



    return 0;

}
