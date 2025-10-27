int ff_wma_run_level_decode(AVCodecContext *avctx, GetBitContext *gb,

                            VLC *vlc, const float *level_table,

                            const uint16_t *run_table, int version,

                            WMACoef *ptr, int offset, int num_coefs,

                            int block_len, int frame_len_bits,

                            int coef_nb_bits)

{

    int code, level, sign;

    const uint32_t *ilvl = (const uint32_t *) level_table;

    uint32_t *iptr = (uint32_t *) ptr;

    const unsigned int coef_mask = block_len - 1;

    for (; offset < num_coefs; offset++) {

        code = get_vlc2(gb, vlc->table, VLCBITS, VLCMAX);

        if (code > 1) {

            /** normal code */

            offset                  += run_table[code];

            sign                     = get_bits1(gb) - 1;

            iptr[offset & coef_mask] = ilvl[code] ^ sign << 31;

        } else if (code == 1) {

            /** EOB */

            break;

        } else {

            /** escape */

            if (!version) {

                level = get_bits(gb, coef_nb_bits);

                /** NOTE: this is rather suboptimal. reading

                 *  block_len_bits would be better */

                offset += get_bits(gb, frame_len_bits);

            } else {

                level = ff_wma_get_large_val(gb);

                /** escape decode */

                if (get_bits1(gb)) {

                    if (get_bits1(gb)) {

                        if (get_bits1(gb)) {

                            av_log(avctx, AV_LOG_ERROR,

                                   "broken escape sequence\n");

                            return -1;

                        } else

                            offset += get_bits(gb, frame_len_bits) + 4;

                    } else

                        offset += get_bits(gb, 2) + 1;

                }

            }

            sign                    = get_bits1(gb) - 1;

            ptr[offset & coef_mask] = (level ^ sign) - sign;

        }

    }

    /** NOTE: EOB can be omitted */

    if (offset > num_coefs) {

        av_log(avctx, AV_LOG_ERROR, "overflow in spectral RLE, ignoring\n");

        return -1;

    }



    return 0;

}
