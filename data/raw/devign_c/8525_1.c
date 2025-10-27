static inline int svq3_decode_block(GetBitContext *gb, DCTELEM *block,

                                    int index, const int type)

{

    static const uint8_t *const scan_patterns[4] =

    { luma_dc_zigzag_scan, zigzag_scan, svq3_scan, chroma_dc_scan };



    int run, level, sign, vlc, limit;

    const int intra           = 3 * type >> 2;

    const uint8_t *const scan = scan_patterns[type];



    for (limit = (16 >> intra); index < 16; index = limit, limit += 8) {

        for (; (vlc = svq3_get_ue_golomb(gb)) != 0; index++) {

            if (vlc == INVALID_VLC)

                return -1;



            sign = (vlc & 0x1) - 1;

            vlc  = vlc + 1 >> 1;



            if (type == 3) {

                if (vlc < 3) {

                    run   = 0;

                    level = vlc;

                } else if (vlc < 4) {

                    run   = 1;

                    level = 1;

                } else {

                    run   = vlc & 0x3;

                    level = (vlc + 9 >> 2) - run;

                }

            } else {

                if (vlc < 16) {

                    run   = svq3_dct_tables[intra][vlc].run;

                    level = svq3_dct_tables[intra][vlc].level;

                } else if (intra) {

                    run   = vlc & 0x7;

                    level = (vlc >> 3) +

                            ((run == 0) ? 8 : ((run < 2) ? 2 : ((run < 5) ? 0 : -1)));

                } else {

                    run   = vlc & 0xF;

                    level = (vlc >> 4) +

                            ((run == 0) ? 4 : ((run < 3) ? 2 : ((run < 10) ? 1 : 0)));

                }

            }



            if ((index += run) >= limit)

                return -1;



            block[scan[index]] = (level ^ sign) - sign;

        }



        if (type != 2) {

            break;

        }

    }



    return 0;

}
