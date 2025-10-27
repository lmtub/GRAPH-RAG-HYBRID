static int colo_packet_compare_common(Packet *ppkt, Packet *spkt, int offset)

{

    if (trace_event_get_state(TRACE_COLO_COMPARE_MISCOMPARE)) {

        char pri_ip_src[20], pri_ip_dst[20], sec_ip_src[20], sec_ip_dst[20];



        strcpy(pri_ip_src, inet_ntoa(ppkt->ip->ip_src));

        strcpy(pri_ip_dst, inet_ntoa(ppkt->ip->ip_dst));

        strcpy(sec_ip_src, inet_ntoa(spkt->ip->ip_src));

        strcpy(sec_ip_dst, inet_ntoa(spkt->ip->ip_dst));



        trace_colo_compare_ip_info(ppkt->size, pri_ip_src,

                                   pri_ip_dst, spkt->size,

                                   sec_ip_src, sec_ip_dst);

    }



    offset = ppkt->vnet_hdr_len + offset;



    if (ppkt->size == spkt->size) {

        return memcmp(ppkt->data + offset,

                      spkt->data + offset,

                      spkt->size - offset);

    } else {

        trace_colo_compare_main("Net packet size are not the same");

        return -1;

    }

}
