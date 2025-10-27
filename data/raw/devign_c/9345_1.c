static int bitplane_decoding(uint8_t* data, int *raw_flag, VC1Context *v)

{

    GetBitContext *gb = &v->s.gb;



    int imode, x, y, code, offset;

    uint8_t invert, *planep = data;

    int width, height, stride;



    width  = v->s.mb_width;

    height = v->s.mb_height >> v->field_mode;

    stride = v->s.mb_stride;

    invert = get_bits1(gb);

    imode = get_vlc2(gb, ff_vc1_imode_vlc.table, VC1_IMODE_VLC_BITS, 1);



    *raw_flag = 0;

    switch (imode) {

    case IMODE_RAW:

        //Data is actually read in the MB layer (same for all tests == "raw")

        *raw_flag = 1; //invert ignored

        return invert;

    case IMODE_DIFF2:

    case IMODE_NORM2:

        if ((height * width) & 1) {

            *planep++ = get_bits1(gb);

            offset    = 1;

        }

        else

            offset = 0;

        // decode bitplane as one long line

        for (y = offset; y < height * width; y += 2) {

            code = get_vlc2(gb, ff_vc1_norm2_vlc.table, VC1_NORM2_VLC_BITS, 1);

            *planep++ = code & 1;

            offset++;

            if (offset == width) {

                offset  = 0;

                planep += stride - width;

            }

            *planep++ = code >> 1;

            offset++;

            if (offset == width) {

                offset  = 0;

                planep += stride - width;

            }

        }

        break;

    case IMODE_DIFF6:

    case IMODE_NORM6:

        if (!(height % 3) && (width % 3)) { // use 2x3 decoding

            for (y = 0; y < height; y += 3) {

                for (x = width & 1; x < width; x += 2) {

                    code = get_vlc2(gb, ff_vc1_norm6_vlc.table, VC1_NORM6_VLC_BITS, 2);

                    if (code < 0) {

                        av_log(v->s.avctx, AV_LOG_DEBUG, "invalid NORM-6 VLC\n");

                        return -1;

                    }

                    planep[x + 0]              = (code >> 0) & 1;

                    planep[x + 1]              = (code >> 1) & 1;

                    planep[x + 0 + stride]     = (code >> 2) & 1;

                    planep[x + 1 + stride]     = (code >> 3) & 1;

                    planep[x + 0 + stride * 2] = (code >> 4) & 1;

                    planep[x + 1 + stride * 2] = (code >> 5) & 1;

                }

                planep += stride * 3;

            }

            if (width & 1)

                decode_colskip(data, 1, height, stride, &v->s.gb);

        } else { // 3x2

            planep += (height & 1) * stride;

            for (y = height & 1; y < height; y += 2) {

                for (x = width % 3; x < width; x += 3) {

                    code = get_vlc2(gb, ff_vc1_norm6_vlc.table, VC1_NORM6_VLC_BITS, 2);

                    if (code < 0) {

                        av_log(v->s.avctx, AV_LOG_DEBUG, "invalid NORM-6 VLC\n");

                        return -1;

                    }

                    planep[x + 0]          = (code >> 0) & 1;

                    planep[x + 1]          = (code >> 1) & 1;

                    planep[x + 2]          = (code >> 2) & 1;

                    planep[x + 0 + stride] = (code >> 3) & 1;

                    planep[x + 1 + stride] = (code >> 4) & 1;

                    planep[x + 2 + stride] = (code >> 5) & 1;

                }

                planep += stride * 2;

            }

            x = width % 3;

            if (x)

                decode_colskip(data,             x, height, stride, &v->s.gb);

            if (height & 1)

                decode_rowskip(data + x, width - x,      1, stride, &v->s.gb);

        }

        break;

    case IMODE_ROWSKIP:

        decode_rowskip(data, width, height, stride, &v->s.gb);

        break;

    case IMODE_COLSKIP:

        decode_colskip(data, width, height, stride, &v->s.gb);

        break;

    default:

        break;

    }



    /* Applying diff operator */

    if (imode == IMODE_DIFF2 || imode == IMODE_DIFF6) {

        planep = data;

        planep[0] ^= invert;

        for (x = 1; x < width; x++)

            planep[x] ^= planep[x-1];

        for (y = 1; y < height; y++) {

            planep += stride;

            planep[0] ^= planep[-stride];

            for (x = 1; x < width; x++) {

                if (planep[x-1] != planep[x-stride]) planep[x] ^= invert;

                else                                 planep[x] ^= planep[x-1];

            }

        }

    } else if (invert) {

        planep = data;

        for (x = 0; x < stride * height; x++)

            planep[x] = !planep[x]; //FIXME stride

    }

    return (imode << 1) + invert;

}
