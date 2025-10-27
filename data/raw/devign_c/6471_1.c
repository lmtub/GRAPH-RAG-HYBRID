static int tscc2_decode_frame(AVCodecContext *avctx, void *data,

                              int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    TSCC2Context *c = avctx->priv_data;

    GetByteContext gb;

    uint32_t frame_type, size;

    int i, val, len, pos = 0;

    int num_mb = c->mb_width * c->mb_height;

    int ret;



    bytestream2_init(&gb, buf, buf_size);

    frame_type = bytestream2_get_byte(&gb);

    if (frame_type > 1) {

        av_log(avctx, AV_LOG_ERROR, "Incorrect frame type %"PRIu32"\n",

               frame_type);

        return AVERROR_INVALIDDATA;

    }



    if ((ret = ff_reget_buffer(avctx, c->pic)) < 0) {

        return ret;

    }



    if (frame_type == 0) {

        *got_frame      = 1;

        if ((ret = av_frame_ref(data, c->pic)) < 0)

            return ret;



        return buf_size;

    }



    if (bytestream2_get_bytes_left(&gb) < 4) {

        av_log(avctx, AV_LOG_ERROR, "Frame is too short\n");

        return AVERROR_INVALIDDATA;

    }



    c->quant[0] = bytestream2_get_byte(&gb);

    c->quant[1] = bytestream2_get_byte(&gb);

    if (c->quant[0] < 2 || c->quant[0] > NUM_VLC_SETS + 1 ||

        c->quant[1] < 2 || c->quant[1] > NUM_VLC_SETS + 1) {

        av_log(avctx, AV_LOG_ERROR, "Invalid quantisers %d / %d\n",

               c->quant[0], c->quant[1]);

        return AVERROR_INVALIDDATA;

    }



    for (i = 0; i < 3; i++) {

        c->q[0][i] = tscc2_quants[c->quant[0] - 2][i];

        c->q[1][i] = tscc2_quants[c->quant[1] - 2][i];

    }



    bytestream2_skip(&gb, 1);



    size = bytestream2_get_le32(&gb);

    if (size > bytestream2_get_bytes_left(&gb)) {

        av_log(avctx, AV_LOG_ERROR, "Slice properties chunk is too large\n");

        return AVERROR_INVALIDDATA;

    }



    for (i = 0; i < size; i++) {

        val   = bytestream2_get_byte(&gb);

        len   = val & 0x3F;

        val >>= 6;

        if (pos + len > num_mb) {

            av_log(avctx, AV_LOG_ERROR, "Too many slice properties\n");

            return AVERROR_INVALIDDATA;

        }

        memset(c->slice_quants + pos, val, len);

        pos += len;

    }

    if (pos < num_mb) {

        av_log(avctx, AV_LOG_ERROR, "Too few slice properties (%d / %d)\n",

               pos, num_mb);

        return AVERROR_INVALIDDATA;

    }



    for (i = 0; i < c->mb_height; i++) {

        size = bytestream2_peek_byte(&gb);

        if (size & 1) {

            size = bytestream2_get_byte(&gb) - 1;

        } else {

            size = bytestream2_get_le32(&gb) >> 1;

        }

        if (!size) {

            int skip_row = 1, j, off = i * c->mb_width;

            for (j = 0; j < c->mb_width; j++) {

                if (c->slice_quants[off + j] == 1 ||

                    c->slice_quants[off + j] == 2) {

                    skip_row = 0;

                    break;

                }

            }

            if (!skip_row) {

                av_log(avctx, AV_LOG_ERROR, "Non-skip row with zero size\n");

                return AVERROR_INVALIDDATA;

            }

        }

        if (bytestream2_get_bytes_left(&gb) < size) {

            av_log(avctx, AV_LOG_ERROR, "Invalid slice size (%"PRIu32"/%u)\n",

                   size, bytestream2_get_bytes_left(&gb));

            return AVERROR_INVALIDDATA;

        }

        ret = tscc2_decode_slice(c, i, buf + bytestream2_tell(&gb), size);

        if (ret) {

            av_log(avctx, AV_LOG_ERROR, "Error decoding slice %d\n", i);

            return ret;

        }

        bytestream2_skip(&gb, size);

    }



    *got_frame      = 1;

    if ((ret = av_frame_ref(data, c->pic)) < 0)

        return ret;



    /* always report that the buffer was completely consumed */

    return buf_size;

}
