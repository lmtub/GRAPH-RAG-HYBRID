static int decode_rle(GetBitContext *gb, uint8_t *pal_dst, int pal_stride,

                      uint8_t *rgb_dst, int rgb_stride, uint32_t *pal,

                      int keyframe, int kf_slipt, int slice, int w, int h)

{

    uint8_t bits[270] = { 0 };

    uint32_t codes[270];

    VLC vlc;



    int current_length = 0, read_codes = 0, next_code = 0, current_codes = 0;

    int remaining_codes, surplus_codes, i;



    const int alphabet_size = 270 - keyframe;



    int last_symbol = 0, repeat = 0, prev_avail = 0;



    if (!keyframe) {

        int x, y, clipw, cliph;



        x     = get_bits(gb, 12);

        y     = get_bits(gb, 12);

        clipw = get_bits(gb, 12) + 1;

        cliph = get_bits(gb, 12) + 1;



        if (x + clipw > w || y + cliph > h)

            return AVERROR_INVALIDDATA;

        pal_dst += pal_stride * y + x;

        rgb_dst += rgb_stride * y + x * 3;

        w        = clipw;

        h        = cliph;

        if (y)

            prev_avail = 1;

    } else {

        if (slice > 0) {

            pal_dst   += pal_stride * kf_slipt;

            rgb_dst   += rgb_stride * kf_slipt;

            prev_avail = 1;

            h         -= kf_slipt;

        } else

            h = kf_slipt;

    }



    /* read explicit codes */

    do {

        while (current_codes--) {

            int symbol = get_bits(gb, 8);

            if (symbol >= 204 - keyframe)

                symbol += 14 - keyframe;

            else if (symbol > 189)

                symbol = get_bits1(gb) + (symbol << 1) - 190;

            if (bits[symbol])

                return AVERROR_INVALIDDATA;

            bits[symbol]  = current_length;

            codes[symbol] = next_code++;

            read_codes++;

        }

        current_length++;

        next_code     <<= 1;

        remaining_codes = (1 << current_length) - next_code;

        current_codes   = get_bits(gb, av_ceil_log2(remaining_codes + 1));

        if (current_length > 22 || current_codes > remaining_codes)

            return AVERROR_INVALIDDATA;

    } while (current_codes != remaining_codes);



    remaining_codes = alphabet_size - read_codes;



    /* determine the minimum length to fit the rest of the alphabet */

    while ((surplus_codes = (2 << current_length) -

                            (next_code << 1) - remaining_codes) < 0) {

        current_length++;

        next_code <<= 1;

    }



    /* add the rest of the symbols lexicographically */

    for (i = 0; i < alphabet_size; i++)

        if (!bits[i]) {

            if (surplus_codes-- == 0) {

                current_length++;

                next_code <<= 1;

            }

            bits[i]  = current_length;

            codes[i] = next_code++;

        }



    if (next_code != 1 << current_length)

        return AVERROR_INVALIDDATA;



    if (i = init_vlc(&vlc, 9, alphabet_size, bits, 1, 1, codes, 4, 4, 0))

        return i;



    /* frame decode */

    do {

        uint8_t *pp = pal_dst;

        uint8_t *rp = rgb_dst;

        do {

            if (repeat-- < 1) {

                int b = get_vlc2(gb, vlc.table, 9, 3);

                if (b < 256)

                    last_symbol = b;

                else if (b < 268) {

                    b -= 256;

                    if (b == 11)

                        b = get_bits(gb, 4) + 10;



                    if (!b)

                        repeat = 0;

                    else

                        repeat = get_bits(gb, b);



                    repeat += (1 << b) - 1;



                    if (last_symbol == -2) {

                        int skip = FFMIN(repeat, pal_dst + w - pp);

                        repeat -= skip;

                        pp     += skip;

                        rp     += skip * 3;

                    }

                } else

                    last_symbol = 267 - b;

            }

            if (last_symbol >= 0) {

                *pp = last_symbol;

                AV_WB24(rp, pal[last_symbol]);

            } else if (last_symbol == -1 && prev_avail) {

                *pp = *(pp - pal_stride);

                memcpy(rp, rp - rgb_stride, 3);

            }

            rp += 3;

        } while (++pp < pal_dst + w);

        pal_dst   += pal_stride;

        rgb_dst   += rgb_stride;

        prev_avail = 1;

    } while (--h);



    ff_free_vlc(&vlc);

    return 0;

}
