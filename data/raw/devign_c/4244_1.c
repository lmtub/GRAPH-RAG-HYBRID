static int kmvc_decode_intra_8x8(KmvcContext * ctx, const uint8_t * src, int src_size, int w, int h)

{

    BitBuf bb;

    int res, val;

    int i, j;

    int bx, by;

    int l0x, l1x, l0y, l1y;

    int mx, my;

    const uint8_t *src_end = src + src_size;



    kmvc_init_getbits(bb, src);



    for (by = 0; by < h; by += 8)

        for (bx = 0; bx < w; bx += 8) {

            kmvc_getbit(bb, src, src_end, res);

            if (!res) {         // fill whole 8x8 block

                if (src >= src_end) {

                    av_log(ctx->avctx, AV_LOG_ERROR, "Data overrun\n");

                    return AVERROR_INVALIDDATA;

                }

                val = *src++;

                for (i = 0; i < 64; i++)

                    BLK(ctx->cur, bx + (i & 0x7), by + (i >> 3)) = val;

            } else {            // handle four 4x4 subblocks

                for (i = 0; i < 4; i++) {

                    l0x = bx + (i & 1) * 4;

                    l0y = by + (i & 2) * 2;

                    kmvc_getbit(bb, src, src_end, res);

                    if (!res) {

                        kmvc_getbit(bb, src, src_end, res);

                        if (!res) {     // fill whole 4x4 block

                            if (src >= src_end) {

                                av_log(ctx->avctx, AV_LOG_ERROR, "Data overrun\n");

                                return AVERROR_INVALIDDATA;

                            }

                            val = *src++;

                            for (j = 0; j < 16; j++)

                                BLK(ctx->cur, l0x + (j & 3), l0y + (j >> 2)) = val;

                        } else {        // copy block from already decoded place

                            if (src >= src_end) {

                                av_log(ctx->avctx, AV_LOG_ERROR, "Data overrun\n");

                                return AVERROR_INVALIDDATA;

                            }

                            val = *src++;

                            mx = val & 0xF;

                            my = val >> 4;

                            for (j = 0; j < 16; j++)

                                BLK(ctx->cur, l0x + (j & 3), l0y + (j >> 2)) =

                                    BLK(ctx->cur, l0x + (j & 3) - mx, l0y + (j >> 2) - my);

                        }

                    } else {    // descend to 2x2 sub-sub-blocks

                        for (j = 0; j < 4; j++) {

                            l1x = l0x + (j & 1) * 2;

                            l1y = l0y + (j & 2);

                            kmvc_getbit(bb, src, src_end, res);

                            if (!res) {

                                kmvc_getbit(bb, src, src_end, res);

                                if (!res) {     // fill whole 2x2 block

                                    if (src >= src_end) {

                                        av_log(ctx->avctx, AV_LOG_ERROR, "Data overrun\n");

                                        return AVERROR_INVALIDDATA;

                                    }

                                    val = *src++;

                                    BLK(ctx->cur, l1x, l1y) = val;

                                    BLK(ctx->cur, l1x + 1, l1y) = val;

                                    BLK(ctx->cur, l1x, l1y + 1) = val;

                                    BLK(ctx->cur, l1x + 1, l1y + 1) = val;

                                } else {        // copy block from already decoded place

                                    if (src >= src_end) {

                                        av_log(ctx->avctx, AV_LOG_ERROR, "Data overrun\n");

                                        return AVERROR_INVALIDDATA;

                                    }

                                    val = *src++;

                                    mx = val & 0xF;

                                    my = val >> 4;

                                    BLK(ctx->cur, l1x, l1y) = BLK(ctx->cur, l1x - mx, l1y - my);

                                    BLK(ctx->cur, l1x + 1, l1y) =

                                        BLK(ctx->cur, l1x + 1 - mx, l1y - my);

                                    BLK(ctx->cur, l1x, l1y + 1) =

                                        BLK(ctx->cur, l1x - mx, l1y + 1 - my);

                                    BLK(ctx->cur, l1x + 1, l1y + 1) =

                                        BLK(ctx->cur, l1x + 1 - mx, l1y + 1 - my);

                                }

                            } else {    // read values for block

                                BLK(ctx->cur, l1x, l1y) = *src++;

                                BLK(ctx->cur, l1x + 1, l1y) = *src++;

                                BLK(ctx->cur, l1x, l1y + 1) = *src++;

                                BLK(ctx->cur, l1x + 1, l1y + 1) = *src++;

                            }

                        }

                    }

                }

            }

        }



    return 0;

}
