static int amr_wb_decode_frame(AVCodecContext *avctx, void *data,

                               int *data_size, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    AMRWBContext *s    = avctx->priv_data;

    int mode;

    int packet_size;

    static const uint8_t block_size[16] = {18, 24, 33, 37, 41, 47, 51, 59, 61, 6, 6, 0, 0, 0, 1, 1};



    mode        = (buf[0] >> 3) & 0x000F;

    packet_size = block_size[mode];



    if (packet_size > buf_size) {

        av_log(avctx, AV_LOG_ERROR, "amr frame too short (%u, should be %u)\n",

               buf_size, packet_size + 1);

        return AVERROR_INVALIDDATA;

    }



    D_IF_decode(s->state, buf, data, _good_frame);

    *data_size = 320 * 2;

    return packet_size;

}
