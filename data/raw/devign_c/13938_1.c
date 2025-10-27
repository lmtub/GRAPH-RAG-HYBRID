static int clv_decode_frame(AVCodecContext *avctx, void *data,

                            int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    CLVContext *c = avctx->priv_data;

    GetByteContext gb;

    uint32_t frame_type;

    int i, j;

    int ret;

    int mb_ret = 0;



    bytestream2_init(&gb, buf, buf_size);

    if (avctx->codec_tag == MKTAG('C','L','V','1')) {

        int skip = bytestream2_get_byte(&gb);

        bytestream2_skip(&gb, (skip + 1) * 8);

    }



    frame_type = bytestream2_get_byte(&gb);

    if ((ret = ff_reget_buffer(avctx, c->pic)) < 0)

        return ret;



    c->pic->key_frame = frame_type & 0x20 ? 1 : 0;

    c->pic->pict_type = frame_type & 0x20 ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;



    if (frame_type & 0x2) {

        if (buf_size < c->mb_width * c->mb_height) {

            av_log(avctx, AV_LOG_ERROR, "Packet too small\n");

            return AVERROR_INVALIDDATA;

        }



        bytestream2_get_be32(&gb); // frame size;

        c->ac_quant        = bytestream2_get_byte(&gb);

        c->luma_dc_quant   = 32;

        c->chroma_dc_quant = 32;



        if ((ret = init_get_bits8(&c->gb, buf + bytestream2_tell(&gb),

                                  (buf_size - bytestream2_tell(&gb)))) < 0)

            return ret;



        for (i = 0; i < 3; i++)

            c->top_dc[i] = 32;

        for (i = 0; i < 4; i++)

            c->left_dc[i] = 32;



        for (j = 0; j < c->mb_height; j++) {

            for (i = 0; i < c->mb_width; i++) {

                ret = decode_mb(c, i, j);

                if (ret < 0)

                    mb_ret = ret;

            }

        }

    } else {

    }



    if ((ret = av_frame_ref(data, c->pic)) < 0)

        return ret;



    *got_frame = 1;



    return mb_ret < 0 ? mb_ret : buf_size;

}
