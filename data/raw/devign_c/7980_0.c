static int binkb_decode_plane(BinkContext *c, AVFrame *frame, BitstreamContext *bc,

                              int plane_idx, int is_key, int is_chroma)

{

    int blk, ret;

    int i, j, bx, by;

    uint8_t *dst, *ref, *ref_start, *ref_end;

    int v, col[2];

    const uint8_t *scan;

    int xoff, yoff;

    LOCAL_ALIGNED_16(int16_t, block, [64]);

    LOCAL_ALIGNED_16(int32_t, dctblock, [64]);

    int coordmap[64];

    int ybias = is_key ? -15 : 0;

    int qp;



    const int stride = frame->linesize[plane_idx];

    int bw = is_chroma ? (c->avctx->width  + 15) >> 4 : (c->avctx->width  + 7) >> 3;

    int bh = is_chroma ? (c->avctx->height + 15) >> 4 : (c->avctx->height + 7) >> 3;



    binkb_init_bundles(c);

    ref_start = frame->data[plane_idx];

    ref_end   = frame->data[plane_idx] + (bh * frame->linesize[plane_idx] + bw) * 8;



    for (i = 0; i < 64; i++)

        coordmap[i] = (i & 7) + (i >> 3) * stride;



    for (by = 0; by < bh; by++) {

        for (i = 0; i < BINKB_NB_SRC; i++) {

            if ((ret = binkb_read_bundle(c, bc, i)) < 0)

                return ret;

        }



        dst  = frame->data[plane_idx]  + 8*by*stride;

        for (bx = 0; bx < bw; bx++, dst += 8) {

            blk = binkb_get_value(c, BINKB_SRC_BLOCK_TYPES);

            switch (blk) {

            case 0:

                break;

            case 1:

                scan = bink_patterns[bitstream_read(bc, 4)];

                i = 0;

                do {

                    int mode = bitstream_read_bit(bc);

                    int run  = bitstream_read(bc, binkb_runbits[i]) + 1;



                    i += run;

                    if (i > 64) {

                        av_log(c->avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                        return AVERROR_INVALIDDATA;

                    }

                    if (mode) {

                        v = binkb_get_value(c, BINKB_SRC_COLORS);

                        for (j = 0; j < run; j++)

                            dst[coordmap[*scan++]] = v;

                    } else {

                        for (j = 0; j < run; j++)

                            dst[coordmap[*scan++]] = binkb_get_value(c, BINKB_SRC_COLORS);

                    }

                } while (i < 63);

                if (i == 63)

                    dst[coordmap[*scan++]] = binkb_get_value(c, BINKB_SRC_COLORS);

                break;

            case 2:

                memset(dctblock, 0, sizeof(*dctblock) * 64);

                dctblock[0] = binkb_get_value(c, BINKB_SRC_INTRA_DC);

                qp = binkb_get_value(c, BINKB_SRC_INTRA_Q);

                read_dct_coeffs(bc, dctblock, bink_scan, binkb_intra_quant, qp);

                c->binkdsp.idct_put(dst, stride, dctblock);

                break;

            case 3:

                xoff = binkb_get_value(c, BINKB_SRC_X_OFF);

                yoff = binkb_get_value(c, BINKB_SRC_Y_OFF) + ybias;

                ref = dst + xoff + yoff * stride;

                if (ref < ref_start || ref + 8*stride > ref_end) {

                    av_log(c->avctx, AV_LOG_WARNING, "Reference block is out of bounds\n");

                } else if (ref + 8*stride < dst || ref >= dst + 8*stride) {

                    c->hdsp.put_pixels_tab[1][0](dst, ref, stride, 8);

                } else {

                    put_pixels8x8_overlapped(dst, ref, stride);

                }

                c->bdsp.clear_block(block);

                v = binkb_get_value(c, BINKB_SRC_INTER_COEFS);

                read_residue(bc, block, v);

                c->binkdsp.add_pixels8(dst, block, stride);

                break;

            case 4:

                xoff = binkb_get_value(c, BINKB_SRC_X_OFF);

                yoff = binkb_get_value(c, BINKB_SRC_Y_OFF) + ybias;

                ref = dst + xoff + yoff * stride;

                if (ref < ref_start || ref + 8 * stride > ref_end) {

                    av_log(c->avctx, AV_LOG_WARNING, "Reference block is out of bounds\n");

                } else if (ref + 8*stride < dst || ref >= dst + 8*stride) {

                    c->hdsp.put_pixels_tab[1][0](dst, ref, stride, 8);

                } else {

                    put_pixels8x8_overlapped(dst, ref, stride);

                }

                memset(dctblock, 0, sizeof(*dctblock) * 64);

                dctblock[0] = binkb_get_value(c, BINKB_SRC_INTER_DC);

                qp = binkb_get_value(c, BINKB_SRC_INTER_Q);

                read_dct_coeffs(bc, dctblock, bink_scan, binkb_inter_quant, qp);

                c->binkdsp.idct_add(dst, stride, dctblock);

                break;

            case 5:

                v = binkb_get_value(c, BINKB_SRC_COLORS);

                c->bdsp.fill_block_tab[1](dst, v, stride, 8);

                break;

            case 6:

                for (i = 0; i < 2; i++)

                    col[i] = binkb_get_value(c, BINKB_SRC_COLORS);

                for (i = 0; i < 8; i++) {

                    v = binkb_get_value(c, BINKB_SRC_PATTERN);

                    for (j = 0; j < 8; j++, v >>= 1)

                        dst[i*stride + j] = col[v & 1];

                }

                break;

            case 7:

                xoff = binkb_get_value(c, BINKB_SRC_X_OFF);

                yoff = binkb_get_value(c, BINKB_SRC_Y_OFF) + ybias;

                ref = dst + xoff + yoff * stride;

                if (ref < ref_start || ref + 8 * stride > ref_end) {

                    av_log(c->avctx, AV_LOG_WARNING, "Reference block is out of bounds\n");

                } else if (ref + 8*stride < dst || ref >= dst + 8*stride) {

                    c->hdsp.put_pixels_tab[1][0](dst, ref, stride, 8);

                } else {

                    put_pixels8x8_overlapped(dst, ref, stride);

                }

                break;

            case 8:

                for (i = 0; i < 8; i++)

                    memcpy(dst + i*stride, c->bundle[BINKB_SRC_COLORS].cur_ptr + i*8, 8);

                c->bundle[BINKB_SRC_COLORS].cur_ptr += 64;

                break;

            default:

                av_log(c->avctx, AV_LOG_ERROR, "Unknown block type %d\n", blk);

                return AVERROR_INVALIDDATA;

            }

        }

    }

    if (bitstream_tell(bc) & 0x1F) // next plane data starts at 32-bit boundary

        bitstream_skip(bc, 32 - (bitstream_tell(bc) & 0x1F));



    return 0;

}
