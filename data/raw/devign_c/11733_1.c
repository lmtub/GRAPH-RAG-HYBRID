MKSCALE16(scale16be, AV_RB16, AV_WB16)

MKSCALE16(scale16le, AV_RL16, AV_WL16)



static int raw_decode(AVCodecContext *avctx, void *data, int *got_frame,

                      AVPacket *avpkt)

{

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(avctx->pix_fmt);

    RawVideoContext *context       = avctx->priv_data;

    const uint8_t *buf             = avpkt->data;

    int buf_size                   = avpkt->size;

    int avpkt_stride               = avpkt->size / avctx->height;

    int linesize_align             = 4;

    int res, len;

    int need_copy;



    AVFrame   *frame   = data;



    if ((avctx->bits_per_coded_sample == 8 || avctx->bits_per_coded_sample == 4

            || avctx->bits_per_coded_sample == 2 || avctx->bits_per_coded_sample == 1) &&

        avctx->pix_fmt == AV_PIX_FMT_PAL8 &&

       (!avctx->codec_tag || avctx->codec_tag == MKTAG('r','a','w',' '))) {

        context->is_1_2_4_8_bpp = 1;

        context->frame_size = av_image_get_buffer_size(avctx->pix_fmt,

                                                       FFALIGN(avctx->width, 16),

                                                       avctx->height, 1);

    } else {

        context->is_lt_16bpp = av_get_bits_per_pixel(desc) == 16 && avctx->bits_per_coded_sample && avctx->bits_per_coded_sample < 16;

        context->frame_size = av_image_get_buffer_size(avctx->pix_fmt, avctx->width,

                                                       avctx->height, 1);

    }

    if (context->frame_size < 0)

        return context->frame_size;



    need_copy = !avpkt->buf || context->is_1_2_4_8_bpp || context->is_yuv2 || context->is_lt_16bpp;



    frame->pict_type        = AV_PICTURE_TYPE_I;

    frame->key_frame        = 1;



    res = ff_decode_frame_props(avctx, frame);

    if (res < 0)

        return res;



    av_frame_set_pkt_pos     (frame, avctx->internal->pkt->pos);

    av_frame_set_pkt_duration(frame, avctx->internal->pkt->duration);



    if (context->tff >= 0) {

        frame->interlaced_frame = 1;

        frame->top_field_first  = context->tff;

    }



    if ((res = av_image_check_size(avctx->width, avctx->height, 0, avctx)) < 0)

        return res;



    if (need_copy)

        frame->buf[0] = av_buffer_alloc(FFMAX(context->frame_size, buf_size));

    else

        frame->buf[0] = av_buffer_ref(avpkt->buf);

    if (!frame->buf[0])

        return AVERROR(ENOMEM);



    // 1 bpp in mov, and 2, 4 and 8 bpp in avi/mov

    if (context->is_1_2_4_8_bpp) {

        int i, j, row_pix = 0;

        uint8_t *dst = frame->buf[0]->data;

        buf_size = context->frame_size - AVPALETTE_SIZE;

        if (avctx->bits_per_coded_sample == 8) {

            for (i = 0, j = 0; j < buf_size && i<avpkt->size; i++, j++) {

                dst[j] = buf[i];

                row_pix++;

                if (row_pix == avctx->width) {

                    i += avpkt_stride - (i % avpkt_stride) - 1;

                    j += 16 - (j % 16) - 1;

                    row_pix = 0;

                }

            }

        } else if (avctx->bits_per_coded_sample == 4) {

            for (i = 0, j = 0; 2 * j + 1 < buf_size && i<avpkt->size; i++, j++) {

                dst[2 * j + 0] = buf[i] >> 4;

                dst[2 * j + 1] = buf[i] & 15;

                row_pix += 2;

                if (row_pix >= avctx->width) {

                    i += avpkt_stride - (i % avpkt_stride) - 1;

                    j += 8 - (j % 8) - 1;

                    row_pix = 0;

                }

            }

        } else if (avctx->bits_per_coded_sample == 2) {

            for (i = 0, j = 0; 4 * j + 3 < buf_size && i<avpkt->size; i++, j++) {

                dst[4 * j + 0] = buf[i] >> 6;

                dst[4 * j + 1] = buf[i] >> 4 & 3;

                dst[4 * j + 2] = buf[i] >> 2 & 3;

                dst[4 * j + 3] = buf[i]      & 3;

                row_pix += 4;

                if (row_pix >= avctx->width) {

                    i += avpkt_stride - (i % avpkt_stride) - 1;

                    j += 4 - (j % 4) - 1;

                    row_pix = 0;

                }

            }

        } else {

            av_assert0(avctx->bits_per_coded_sample == 1);

            for (i = 0, j = 0; 8 * j + 7 < buf_size && i<avpkt->size; i++, j++) {

                dst[8 * j + 0] = buf[i] >> 7;

                dst[8 * j + 1] = buf[i] >> 6 & 1;

                dst[8 * j + 2] = buf[i] >> 5 & 1;

                dst[8 * j + 3] = buf[i] >> 4 & 1;

                dst[8 * j + 4] = buf[i] >> 3 & 1;

                dst[8 * j + 5] = buf[i] >> 2 & 1;

                dst[8 * j + 6] = buf[i] >> 1 & 1;

                dst[8 * j + 7] = buf[i]      & 1;

                row_pix += 8;

                if (row_pix >= avctx->width) {

                    i += avpkt_stride - (i % avpkt_stride) - 1;

                    j += 2 - (j % 2) - 1;

                    row_pix = 0;

                }

            }

        }

        linesize_align = 16;

        buf = dst;

    } else if (context->is_lt_16bpp) {

        uint8_t *dst = frame->buf[0]->data;

        int packed = (avctx->codec_tag & 0xFFFFFF) == MKTAG('B','I','T', 0);

        int swap   =  avctx->codec_tag >> 24;



        if (packed && swap) {

            av_fast_padded_malloc(&context->bitstream_buf, &context->bitstream_buf_size, buf_size);

            if (!context->bitstream_buf)

                return AVERROR(ENOMEM);

            if (swap == 16)

                context->bbdsp.bswap16_buf(context->bitstream_buf, (const uint16_t*)buf, buf_size / 2);

            else if (swap == 32)

                context->bbdsp.bswap_buf(context->bitstream_buf, (const uint32_t*)buf, buf_size / 4);

            else

                return AVERROR_INVALIDDATA;

            buf = context->bitstream_buf;

        }



        if (desc->flags & AV_PIX_FMT_FLAG_BE)

            scale16be(avctx, dst, buf, buf_size, packed);

        else

            scale16le(avctx, dst, buf, buf_size, packed);



        buf = dst;

    } else if (need_copy) {

        memcpy(frame->buf[0]->data, buf, buf_size);

        buf = frame->buf[0]->data;

    }



    if (avctx->codec_tag == MKTAG('A', 'V', '1', 'x') ||

        avctx->codec_tag == MKTAG('A', 'V', 'u', 'p'))

        buf += buf_size - context->frame_size;



    len = context->frame_size - (avctx->pix_fmt==AV_PIX_FMT_PAL8 ? AVPALETTE_SIZE : 0);

    if (buf_size < len && ((avctx->codec_tag & 0xFFFFFF) != MKTAG('B','I','T', 0) || !need_copy)) {

        av_log(avctx, AV_LOG_ERROR, "Invalid buffer size, packet size %d < expected frame_size %d\n", buf_size, len);

        av_buffer_unref(&frame->buf[0]);

        return AVERROR(EINVAL);

    }



    if ((res = av_image_fill_arrays(frame->data, frame->linesize,

                                    buf, avctx->pix_fmt,

                                    avctx->width, avctx->height, 1)) < 0) {

        av_buffer_unref(&frame->buf[0]);

        return res;

    }



    if (avctx->pix_fmt == AV_PIX_FMT_PAL8) {

        const uint8_t *pal = av_packet_get_side_data(avpkt, AV_PKT_DATA_PALETTE,

                                                     NULL);



        if (pal) {

            av_buffer_unref(&context->palette);

            context->palette = av_buffer_alloc(AVPALETTE_SIZE);

            if (!context->palette) {

                av_buffer_unref(&frame->buf[0]);

                return AVERROR(ENOMEM);

            }

            memcpy(context->palette->data, pal, AVPALETTE_SIZE);

            frame->palette_has_changed = 1;

        }

    }



    if ((avctx->pix_fmt==AV_PIX_FMT_BGR24    ||

        avctx->pix_fmt==AV_PIX_FMT_GRAY8    ||

        avctx->pix_fmt==AV_PIX_FMT_RGB555LE ||

        avctx->pix_fmt==AV_PIX_FMT_RGB555BE ||

        avctx->pix_fmt==AV_PIX_FMT_RGB565LE ||

        avctx->pix_fmt==AV_PIX_FMT_MONOWHITE ||

        avctx->pix_fmt==AV_PIX_FMT_PAL8) &&

        FFALIGN(frame->linesize[0], linesize_align) * avctx->height <= buf_size)

        frame->linesize[0] = FFALIGN(frame->linesize[0], linesize_align);



    if (avctx->pix_fmt == AV_PIX_FMT_NV12 && avctx->codec_tag == MKTAG('N', 'V', '1', '2') &&

        FFALIGN(frame->linesize[0], linesize_align) * avctx->height +

        FFALIGN(frame->linesize[1], linesize_align) * ((avctx->height + 1) / 2) <= buf_size) {

        int la0 = FFALIGN(frame->linesize[0], linesize_align);

        frame->data[1] += (la0 - frame->linesize[0]) * avctx->height;

        frame->linesize[0] = la0;

        frame->linesize[1] = FFALIGN(frame->linesize[1], linesize_align);

    }



    if ((avctx->pix_fmt == AV_PIX_FMT_PAL8 && buf_size < context->frame_size) ||

        (desc->flags & AV_PIX_FMT_FLAG_PSEUDOPAL)) {

        frame->buf[1]  = av_buffer_ref(context->palette);

        if (!frame->buf[1]) {

            av_buffer_unref(&frame->buf[0]);

            return AVERROR(ENOMEM);

        }

        frame->data[1] = frame->buf[1]->data;

    }



    if (avctx->pix_fmt == AV_PIX_FMT_BGR24 &&

        ((frame->linesize[0] + 3) & ~3) * avctx->height <= buf_size)

        frame->linesize[0] = (frame->linesize[0] + 3) & ~3;



    if (context->flip)

        flip(avctx, frame);



    if (avctx->codec_tag == MKTAG('Y', 'V', '1', '2') ||

        avctx->codec_tag == MKTAG('Y', 'V', '1', '6') ||

        avctx->codec_tag == MKTAG('Y', 'V', '2', '4') ||

        avctx->codec_tag == MKTAG('Y', 'V', 'U', '9'))

        FFSWAP(uint8_t *, frame->data[1], frame->data[2]);



    if (avctx->codec_tag == AV_RL32("I420") && (avctx->width+1)*(avctx->height+1) * 3/2 == buf_size) {

        frame->data[1] = frame->data[1] +  (avctx->width+1)*(avctx->height+1) -avctx->width*avctx->height;

        frame->data[2] = frame->data[2] + ((avctx->width+1)*(avctx->height+1) -avctx->width*avctx->height)*5/4;

    }



    if (avctx->codec_tag == AV_RL32("yuv2") &&

        avctx->pix_fmt   == AV_PIX_FMT_YUYV422) {

        int x, y;

        uint8_t *line = frame->data[0];

        for (y = 0; y < avctx->height; y++) {

            for (x = 0; x < avctx->width; x++)

                line[2 * x + 1] ^= 0x80;

            line += frame->linesize[0];

        }

    }



    if (avctx->field_order > AV_FIELD_PROGRESSIVE) { /* we have interlaced material flagged in container */

        frame->interlaced_frame = 1;

        if (avctx->field_order == AV_FIELD_TT || avctx->field_order == AV_FIELD_TB)

            frame->top_field_first = 1;

    }



    *got_frame = 1;

    return buf_size;

}
