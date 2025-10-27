int ff_alloc_packet(AVPacket *avpkt, int size)

{

    if (size > INT_MAX - FF_INPUT_BUFFER_PADDING_SIZE)

        return AVERROR(EINVAL);



    if (avpkt->data) {

        void *destruct = avpkt->destruct;



        if (avpkt->size < size)

            return AVERROR(EINVAL);



        av_init_packet(avpkt);

        avpkt->destruct = destruct;

        avpkt->size = size;

        return 0;

    } else {

        return av_new_packet(avpkt, size);

    }

}
