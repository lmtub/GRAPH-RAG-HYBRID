int av_new_packet(AVPacket *pkt, int size)

{

    uint8_t *data;

    if((unsigned)size > (unsigned)size + FF_INPUT_BUFFER_PADDING_SIZE)

        return AVERROR(ENOMEM);

    data = av_malloc(size + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!data)

        return AVERROR(ENOMEM);

    memset(data + size, 0, FF_INPUT_BUFFER_PADDING_SIZE);



    av_init_packet(pkt);

    pkt->data = data;

    pkt->size = size;

    pkt->destruct = av_destruct_packet;

    return 0;

}
