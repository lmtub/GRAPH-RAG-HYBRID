static int srt_decode_frame(AVCodecContext *avctx,

                            void *data, int *got_sub_ptr, AVPacket *avpkt)

{

    AVSubtitle *sub = data;

    AVBPrint buffer;

    int x1 = -1, y1 = -1, x2 = -1, y2 = -1;

    int size, ret;

    const uint8_t *p = av_packet_get_side_data(avpkt, AV_PKT_DATA_SUBTITLE_POSITION, &size);

    FFASSDecoderContext *s = avctx->priv_data;



    if (p && size == 16) {

        x1 = AV_RL32(p     );

        y1 = AV_RL32(p +  4);

        x2 = AV_RL32(p +  8);

        y2 = AV_RL32(p + 12);

    }



    if (avpkt->size <= 0)

        return avpkt->size;



    av_bprint_init(&buffer, 0, AV_BPRINT_SIZE_UNLIMITED);



    srt_to_ass(avctx, &buffer, avpkt->data, x1, y1, x2, y2);

    ret = ff_ass_add_rect(sub, buffer.str, s->readorder++, 0, NULL, NULL);

    av_bprint_finalize(&buffer, NULL);

    if (ret < 0)

        return ret;



    *got_sub_ptr = sub->num_rects > 0;

    return avpkt->size;

}
