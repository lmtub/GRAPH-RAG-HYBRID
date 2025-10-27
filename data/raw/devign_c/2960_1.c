void av_destruct_packet(AVPacket *pkt)

{

    int i;



    av_free(pkt->data);

    pkt->data = NULL; pkt->size = 0;



    for (i = 0; i < pkt->side_data_elems; i++)

        av_free(pkt->side_data[i].data);

    av_freep(&pkt->side_data);

    pkt->side_data_elems = 0;

}
