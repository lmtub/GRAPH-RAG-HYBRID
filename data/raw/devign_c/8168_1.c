uint8_t *av_packet_new_side_data(AVPacket *pkt, enum AVPacketSideDataType type,

                                 int size)

{

    int elems = pkt->side_data_elems;



    if ((unsigned)elems + 1 > INT_MAX / sizeof(*pkt->side_data))

        return NULL;

    if ((unsigned)size > INT_MAX - FF_INPUT_BUFFER_PADDING_SIZE)

        return NULL;



    pkt->side_data = av_realloc(pkt->side_data,

                                (elems + 1) * sizeof(*pkt->side_data));

    if (!pkt->side_data)

        return NULL;



    pkt->side_data[elems].data = av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!pkt->side_data[elems].data)

        return NULL;

    pkt->side_data[elems].size = size;

    pkt->side_data[elems].type = type;

    pkt->side_data_elems++;



    return pkt->side_data[elems].data;

}
