static av_always_inline void xchg_mb_border(H264Context *h, H264SliceContext *sl,

                                            uint8_t *src_y,

                                            uint8_t *src_cb, uint8_t *src_cr,

                                            int linesize, int uvlinesize,

                                            int xchg, int chroma444,

                                            int simple, int pixel_shift)

{

    int deblock_topleft;

    int deblock_top;

    int top_idx = 1;

    uint8_t *top_border_m1;

    uint8_t *top_border;



    if (!simple && FRAME_MBAFF(h)) {

        if (h->mb_y & 1) {

            if (!MB_MBAFF(h))

                return;

        } else {

            top_idx = MB_MBAFF(h) ? 0 : 1;

        }

    }



    if (h->deblocking_filter == 2) {

        deblock_topleft = h->slice_table[h->mb_xy - 1 - h->mb_stride] == sl->slice_num;

        deblock_top     = sl->top_type;

    } else {

        deblock_topleft = (h->mb_x > 0);

        deblock_top     = (h->mb_y > !!MB_FIELD(h));

    }



    src_y  -= linesize   + 1 + pixel_shift;

    src_cb -= uvlinesize + 1 + pixel_shift;

    src_cr -= uvlinesize + 1 + pixel_shift;



    top_border_m1 = h->top_borders[top_idx][h->mb_x - 1];

    top_border    = h->top_borders[top_idx][h->mb_x];



#define XCHG(a, b, xchg)                        \

    if (pixel_shift) {                          \

        if (xchg) {                             \

            AV_SWAP64(b + 0, a + 0);            \

            AV_SWAP64(b + 8, a + 8);            \

        } else {                                \

            AV_COPY128(b, a);                   \

        }                                       \

    } else if (xchg)                            \

        AV_SWAP64(b, a);                        \

    else                                        \

        AV_COPY64(b, a);



    if (deblock_top) {

        if (deblock_topleft) {

            XCHG(top_border_m1 + (8 << pixel_shift),

                 src_y - (7 << pixel_shift), 1);

        }

        XCHG(top_border + (0 << pixel_shift), src_y + (1 << pixel_shift), xchg);

        XCHG(top_border + (8 << pixel_shift), src_y + (9 << pixel_shift), 1);

        if (h->mb_x + 1 < h->mb_width) {

            XCHG(h->top_borders[top_idx][h->mb_x + 1],

                 src_y + (17 << pixel_shift), 1);

        }

    }

    if (simple || !CONFIG_GRAY || !(h->flags & CODEC_FLAG_GRAY)) {

        if (chroma444) {

            if (deblock_top) {

                if (deblock_topleft) {

                    XCHG(top_border_m1 + (24 << pixel_shift), src_cb - (7 << pixel_shift), 1);

                    XCHG(top_border_m1 + (40 << pixel_shift), src_cr - (7 << pixel_shift), 1);

                }

                XCHG(top_border + (16 << pixel_shift), src_cb + (1 << pixel_shift), xchg);

                XCHG(top_border + (24 << pixel_shift), src_cb + (9 << pixel_shift), 1);

                XCHG(top_border + (32 << pixel_shift), src_cr + (1 << pixel_shift), xchg);

                XCHG(top_border + (40 << pixel_shift), src_cr + (9 << pixel_shift), 1);

                if (h->mb_x + 1 < h->mb_width) {

                    XCHG(h->top_borders[top_idx][h->mb_x + 1] + (16 << pixel_shift), src_cb + (17 << pixel_shift), 1);

                    XCHG(h->top_borders[top_idx][h->mb_x + 1] + (32 << pixel_shift), src_cr + (17 << pixel_shift), 1);

                }

            }

        } else {

            if (deblock_top) {

                if (deblock_topleft) {

                    XCHG(top_border_m1 + (16 << pixel_shift), src_cb - (7 << pixel_shift), 1);

                    XCHG(top_border_m1 + (24 << pixel_shift), src_cr - (7 << pixel_shift), 1);

                }

                XCHG(top_border + (16 << pixel_shift), src_cb + 1 + pixel_shift, 1);

                XCHG(top_border + (24 << pixel_shift), src_cr + 1 + pixel_shift, 1);

            }

        }

    }

}
