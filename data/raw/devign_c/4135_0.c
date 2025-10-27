static int colo_packet_compare(Packet *ppkt, Packet *spkt)

{

    trace_colo_compare_ip_info(ppkt->size, inet_ntoa(ppkt->ip->ip_src),

                               inet_ntoa(ppkt->ip->ip_dst), spkt->size,

                               inet_ntoa(spkt->ip->ip_src),

                               inet_ntoa(spkt->ip->ip_dst));



    if (ppkt->size == spkt->size) {

        return memcmp(ppkt->data, spkt->data, spkt->size);

    } else {

        return -1;

    }

}
