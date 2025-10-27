static int mss2_decode_frame(AVCodecContext *avctx, void *data, int *got_frame,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    MSS2Context *ctx = avctx->priv_data;

    MSS12Context *c  = &ctx->c;

    AVFrame *frame   = data;

    GetBitContext gb;

    GetByteContext gB;

    ArithCoder acoder;



    int keyframe, has_wmv9, has_mv, is_rle, is_555, ret;



    Rectangle wmv9rects[MAX_WMV9_RECTANGLES], *r;

    int used_rects = 0, i, implicit_rect = 0, av_uninit(wmv9_mask);



    if ((ret = init_get_bits8(&gb, buf, buf_size)) < 0)

        return ret;



    if (keyframe = get_bits1(&gb))

        skip_bits(&gb, 7);

    has_wmv9 = get_bits1(&gb);

    has_mv   = keyframe ? 0 : get_bits1(&gb);

    is_rle   = get_bits1(&gb);

    is_555   = is_rle && get_bits1(&gb);

    if (c->slice_split > 0)

        ctx->split_position = c->slice_split;

    else if (c->slice_split < 0) {

        if (get_bits1(&gb)) {

            if (get_bits1(&gb)) {

                if (get_bits1(&gb))

                    ctx->split_position = get_bits(&gb, 16);

                else

                    ctx->split_position = get_bits(&gb, 12);

            } else

                ctx->split_position = get_bits(&gb, 8) << 4;

        } else {

            if (keyframe)

                ctx->split_position = avctx->height / 2;

        }

    } else

        ctx->split_position = avctx->height;



    if (c->slice_split && (ctx->split_position < 1 - is_555 ||

                           ctx->split_position > avctx->height - 1))

        return AVERROR_INVALIDDATA;



    align_get_bits(&gb);

    buf      += get_bits_count(&gb) >> 3;

    buf_size -= get_bits_count(&gb) >> 3;



    if (buf_size < 1)

        return AVERROR_INVALIDDATA;



    if (is_555 && (has_wmv9 || has_mv || c->slice_split && ctx->split_position))

        return AVERROR_INVALIDDATA;



    avctx->pix_fmt = is_555 ? AV_PIX_FMT_RGB555 : AV_PIX_FMT_RGB24;

    if (ctx->last_pic->format != avctx->pix_fmt)

        av_frame_unref(ctx->last_pic);



    if (has_wmv9) {

        bytestream2_init(&gB, buf, buf_size + ARITH2_PADDING);

        arith2_init(&acoder, &gB);



        implicit_rect = !arith2_get_bit(&acoder);



        while (arith2_get_bit(&acoder)) {

            if (used_rects == MAX_WMV9_RECTANGLES)

                return AVERROR_INVALIDDATA;

            r = &wmv9rects[used_rects];

            if (!used_rects)

                r->x = arith2_get_number(&acoder, avctx->width);

            else

                r->x = arith2_get_number(&acoder, avctx->width -

                                         wmv9rects[used_rects - 1].x) +

                       wmv9rects[used_rects - 1].x;

            r->y = arith2_get_number(&acoder, avctx->height);

            r->w = arith2_get_number(&acoder, avctx->width  - r->x) + 1;

            r->h = arith2_get_number(&acoder, avctx->height - r->y) + 1;

            used_rects++;

        }



        if (implicit_rect && used_rects) {

            av_log(avctx, AV_LOG_ERROR, "implicit_rect && used_rects > 0\n");

            return AVERROR_INVALIDDATA;

        }



        if (implicit_rect) {

            wmv9rects[0].x = 0;

            wmv9rects[0].y = 0;

            wmv9rects[0].w = avctx->width;

            wmv9rects[0].h = avctx->height;



            used_rects = 1;

        }

        for (i = 0; i < used_rects; i++) {

            if (!implicit_rect && arith2_get_bit(&acoder)) {

                av_log(avctx, AV_LOG_ERROR, "Unexpected grandchildren\n");

                return AVERROR_INVALIDDATA;

            }

            if (!i) {

                wmv9_mask = arith2_get_bit(&acoder) - 1;

                if (!wmv9_mask)

                    wmv9_mask = arith2_get_number(&acoder, 256);

            }

            wmv9rects[i].coded = arith2_get_number(&acoder, 2);

        }



        buf      += arith2_get_consumed_bytes(&acoder);

        buf_size -= arith2_get_consumed_bytes(&acoder);

        if (buf_size < 1)

            return AVERROR_INVALIDDATA;

    }



    c->mvX = c->mvY = 0;

    if (keyframe && !is_555) {

        if ((i = decode_pal_v2(c, buf, buf_size)) < 0)

            return AVERROR_INVALIDDATA;

        buf      += i;

        buf_size -= i;

    } else if (has_mv) {

        buf      += 4;

        buf_size -= 4;

        if (buf_size < 1)

            return AVERROR_INVALIDDATA;

        c->mvX = AV_RB16(buf - 4) - avctx->width;

        c->mvY = AV_RB16(buf - 2) - avctx->height;

    }



    if (c->mvX < 0 || c->mvY < 0) {

        FFSWAP(uint8_t *, c->pal_pic, c->last_pal_pic);



        if ((ret = ff_get_buffer(avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)

            return ret;



        if (ctx->last_pic->data[0]) {

            av_assert0(frame->linesize[0] == ctx->last_pic->linesize[0]);

            c->last_rgb_pic = ctx->last_pic->data[0] +

                              ctx->last_pic->linesize[0] * (avctx->height - 1);

        } else {

            av_log(avctx, AV_LOG_ERROR, "Missing keyframe\n");

            return AVERROR_INVALIDDATA;

        }

    } else {

        if ((ret = ff_reget_buffer(avctx, ctx->last_pic)) < 0)

            return ret;

        if ((ret = av_frame_ref(frame, ctx->last_pic)) < 0)

            return ret;



        c->last_rgb_pic = NULL;

    }

    c->rgb_pic    = frame->data[0] +

                    frame->linesize[0] * (avctx->height - 1);

    c->rgb_stride = -frame->linesize[0];



    frame->key_frame = keyframe;

    frame->pict_type = keyframe ? AV_PICTURE_TYPE_I : AV_PICTURE_TYPE_P;



    if (is_555) {

        bytestream2_init(&gB, buf, buf_size);



        if (decode_555(&gB, (uint16_t *)c->rgb_pic, c->rgb_stride >> 1,

                       keyframe, avctx->width, avctx->height))

            return AVERROR_INVALIDDATA;



        buf_size -= bytestream2_tell(&gB);

    } else {

        if (keyframe) {

            c->corrupted = 0;

            ff_mss12_slicecontext_reset(&ctx->sc[0]);

            if (c->slice_split)

                ff_mss12_slicecontext_reset(&ctx->sc[1]);

        }

        if (is_rle) {

            if ((ret = init_get_bits8(&gb, buf, buf_size)) < 0)

                return ret;

            if (ret = decode_rle(&gb, c->pal_pic, c->pal_stride,

                                 c->rgb_pic, c->rgb_stride, c->pal, keyframe,

                                 ctx->split_position, 0,

                                 avctx->width, avctx->height))

                return ret;

            align_get_bits(&gb);



            if (c->slice_split)

                if (ret = decode_rle(&gb, c->pal_pic, c->pal_stride,

                                     c->rgb_pic, c->rgb_stride, c->pal, keyframe,

                                     ctx->split_position, 1,

                                     avctx->width, avctx->height))

                    return ret;



            align_get_bits(&gb);

            buf      += get_bits_count(&gb) >> 3;

            buf_size -= get_bits_count(&gb) >> 3;

        } else if (!implicit_rect || wmv9_mask != -1) {

            if (c->corrupted)

                return AVERROR_INVALIDDATA;

            bytestream2_init(&gB, buf, buf_size + ARITH2_PADDING);

            arith2_init(&acoder, &gB);

            c->keyframe = keyframe;

            if (c->corrupted = ff_mss12_decode_rect(&ctx->sc[0], &acoder, 0, 0,

                                                    avctx->width,

                                                    ctx->split_position))

                return AVERROR_INVALIDDATA;



            buf      += arith2_get_consumed_bytes(&acoder);

            buf_size -= arith2_get_consumed_bytes(&acoder);

            if (c->slice_split) {

                if (buf_size < 1)

                    return AVERROR_INVALIDDATA;

                bytestream2_init(&gB, buf, buf_size + ARITH2_PADDING);

                arith2_init(&acoder, &gB);

                if (c->corrupted = ff_mss12_decode_rect(&ctx->sc[1], &acoder, 0,

                                                        ctx->split_position,

                                                        avctx->width,

                                                        avctx->height - ctx->split_position))

                    return AVERROR_INVALIDDATA;



                buf      += arith2_get_consumed_bytes(&acoder);

                buf_size -= arith2_get_consumed_bytes(&acoder);

            }

        } else

            memset(c->pal_pic, 0, c->pal_stride * avctx->height);

    }



    if (has_wmv9) {

        for (i = 0; i < used_rects; i++) {

            int x = wmv9rects[i].x;

            int y = wmv9rects[i].y;

            int w = wmv9rects[i].w;

            int h = wmv9rects[i].h;

            if (wmv9rects[i].coded) {

                int WMV9codedFrameSize;

                if (buf_size < 4 || !(WMV9codedFrameSize = AV_RL24(buf)))

                    return AVERROR_INVALIDDATA;

                if (ret = decode_wmv9(avctx, buf + 3, buf_size - 3,

                                      x, y, w, h, wmv9_mask))

                    return ret;

                buf      += WMV9codedFrameSize + 3;

                buf_size -= WMV9codedFrameSize + 3;

            } else {

                uint8_t *dst = c->rgb_pic + y * c->rgb_stride + x * 3;

                if (wmv9_mask != -1) {

                    ctx->dsp.mss2_gray_fill_masked(dst, c->rgb_stride,

                                                   wmv9_mask,

                                                   c->pal_pic + y * c->pal_stride + x,

                                                   c->pal_stride,

                                                   w, h);

                } else {

                    do {

                        memset(dst, 0x80, w * 3);

                        dst += c->rgb_stride;

                    } while (--h);

                }

            }

        }

    }



    if (buf_size)

        av_log(avctx, AV_LOG_WARNING, "buffer not fully consumed\n");



    if (c->mvX < 0 || c->mvY < 0) {

        av_frame_unref(ctx->last_pic);

        ret = av_frame_ref(ctx->last_pic, frame);

        if (ret < 0)

            return ret;

    }



    *got_frame       = 1;



    return avpkt->size;

}
