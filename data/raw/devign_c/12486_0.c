static int webp_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                             AVPacket *avpkt)

{

    AVFrame * const p = data;

    WebPContext *s = avctx->priv_data;

    GetByteContext gb;

    int ret;

    uint32_t chunk_type, chunk_size;

    int vp8x_flags = 0;



    s->avctx     = avctx;

    s->width     = 0;

    s->height    = 0;

    *got_frame   = 0;

    s->has_alpha = 0;

    bytestream2_init(&gb, avpkt->data, avpkt->size);



    if (bytestream2_get_bytes_left(&gb) < 12)

        return AVERROR_INVALIDDATA;



    if (bytestream2_get_le32(&gb) != MKTAG('R', 'I', 'F', 'F')) {

        av_log(avctx, AV_LOG_ERROR, "missing RIFF tag\n");

        return AVERROR_INVALIDDATA;

    }



    chunk_size = bytestream2_get_le32(&gb);

    if (bytestream2_get_bytes_left(&gb) < chunk_size)

        return AVERROR_INVALIDDATA;



    if (bytestream2_get_le32(&gb) != MKTAG('W', 'E', 'B', 'P')) {

        av_log(avctx, AV_LOG_ERROR, "missing WEBP tag\n");

        return AVERROR_INVALIDDATA;

    }



    while (bytestream2_get_bytes_left(&gb) > 0) {

        char chunk_str[5] = { 0 };



        chunk_type = bytestream2_get_le32(&gb);

        chunk_size = bytestream2_get_le32(&gb);

        if (chunk_size == UINT32_MAX)

            return AVERROR_INVALIDDATA;

        chunk_size += chunk_size & 1;



        if (bytestream2_get_bytes_left(&gb) < chunk_size)

            return AVERROR_INVALIDDATA;



        switch (chunk_type) {

        case MKTAG('V', 'P', '8', ' '):

            if (!*got_frame) {

                ret = vp8_lossy_decode_frame(avctx, p, got_frame,

                                             avpkt->data + bytestream2_tell(&gb),

                                             chunk_size);

                if (ret < 0)

                    return ret;

            }

            bytestream2_skip(&gb, chunk_size);

            break;

        case MKTAG('V', 'P', '8', 'L'):

            if (!*got_frame) {

                ret = vp8_lossless_decode_frame(avctx, p, got_frame,

                                                avpkt->data + bytestream2_tell(&gb),

                                                chunk_size, 0);

                if (ret < 0)

                    return ret;

            }

            bytestream2_skip(&gb, chunk_size);

            break;

        case MKTAG('V', 'P', '8', 'X'):

            vp8x_flags = bytestream2_get_byte(&gb);

            bytestream2_skip(&gb, 3);

            s->width  = bytestream2_get_le24(&gb) + 1;

            s->height = bytestream2_get_le24(&gb) + 1;

            ret = av_image_check_size(s->width, s->height, 0, avctx);

            if (ret < 0)

                return ret;

            break;

        case MKTAG('A', 'L', 'P', 'H'): {

            int alpha_header, filter_m, compression;



            if (!(vp8x_flags & VP8X_FLAG_ALPHA)) {

                av_log(avctx, AV_LOG_WARNING,

                       "ALPHA chunk present, but alpha bit not set in the "

                       "VP8X header\n");

            }

            if (chunk_size == 0) {

                av_log(avctx, AV_LOG_ERROR, "invalid ALPHA chunk size\n");

                return AVERROR_INVALIDDATA;

            }

            alpha_header       = bytestream2_get_byte(&gb);

            s->alpha_data      = avpkt->data + bytestream2_tell(&gb);

            s->alpha_data_size = chunk_size - 1;

            bytestream2_skip(&gb, s->alpha_data_size);



            filter_m    = (alpha_header >> 2) & 0x03;

            compression =  alpha_header       & 0x03;



            if (compression > ALPHA_COMPRESSION_VP8L) {

                av_log(avctx, AV_LOG_VERBOSE,

                       "skipping unsupported ALPHA chunk\n");

            } else {

                s->has_alpha         = 1;

                s->alpha_compression = compression;

                s->alpha_filter      = filter_m;

            }



            break;

        }

        case MKTAG('I', 'C', 'C', 'P'):

        case MKTAG('A', 'N', 'I', 'M'):

        case MKTAG('A', 'N', 'M', 'F'):

        case MKTAG('E', 'X', 'I', 'F'):

        case MKTAG('X', 'M', 'P', ' '):

            AV_WL32(chunk_str, chunk_type);

            av_log(avctx, AV_LOG_VERBOSE, "skipping unsupported chunk: %s\n",

                   chunk_str);

            bytestream2_skip(&gb, chunk_size);

            break;

        default:

            AV_WL32(chunk_str, chunk_type);

            av_log(avctx, AV_LOG_VERBOSE, "skipping unknown chunk: %s\n",

                   chunk_str);

            bytestream2_skip(&gb, chunk_size);

            break;

        }

    }



    if (!*got_frame) {

        av_log(avctx, AV_LOG_ERROR, "image data not found\n");

        return AVERROR_INVALIDDATA;

    }



    return avpkt->size;

}
