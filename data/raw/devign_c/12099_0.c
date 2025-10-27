static int cinaudio_decode_frame(AVCodecContext *avctx,

                                 void *data, int *data_size,

                                 AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    CinAudioContext *cin = avctx->priv_data;

    const uint8_t *src = buf;

    int16_t *samples = data;

    int delta;



    buf_size = FFMIN(buf_size, *data_size/2);



    delta = cin->delta;

    if (cin->initial_decode_frame) {

        cin->initial_decode_frame = 0;

        delta = (int16_t)AV_RL16(src); src += 2;

        *samples++ = delta;

        buf_size -= 2;

    }

    while (buf_size > 0) {

        delta += cinaudio_delta16_table[*src++];

        delta = av_clip_int16(delta);

        *samples++ = delta;

        --buf_size;

    }

    cin->delta = delta;



    *data_size = (uint8_t *)samples - (uint8_t *)data;



    return src - buf;

}
