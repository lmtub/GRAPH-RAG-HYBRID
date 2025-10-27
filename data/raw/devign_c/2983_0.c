static int colo_packet_compare_other(Packet *spkt, Packet *ppkt)

{

    trace_colo_compare_main("compare other");

    trace_colo_compare_ip_info(ppkt->size, inet_ntoa(ppkt->ip->ip_src),

                               inet_ntoa(ppkt->ip->ip_dst), spkt->size,

                               inet_ntoa(spkt->ip->ip_src),

                               inet_ntoa(spkt->ip->ip_dst));

    return colo_packet_compare(ppkt, spkt);

}
