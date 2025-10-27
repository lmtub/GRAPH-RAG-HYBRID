static av_always_inline void FUNC(intra_pred)(HEVCContext *s, int x0, int y0,

                                              int log2_size, int c_idx)

{

#define PU(x) \

    ((x) >> s->ps.sps->log2_min_pu_size)

#define MVF(x, y) \

    (s->ref->tab_mvf[(x) + (y) * min_pu_width])

#define MVF_PU(x, y) \

    MVF(PU(x0 + ((x) << hshift)), PU(y0 + ((y) << vshift)))

#define IS_INTRA(x, y) \

    (MVF_PU(x, y).pred_flag == PF_INTRA)

#define MIN_TB_ADDR_ZS(x, y) \

    s->ps.pps->min_tb_addr_zs[(y) * (s->ps.sps->tb_mask+2) + (x)]

#define EXTEND(ptr, val, len)         \

do {                                  \

    pixel4 pix = PIXEL_SPLAT_X4(val); \

    for (i = 0; i < (len); i += 4)    \

        AV_WN4P(ptr + i, pix);        \

} while (0)



#define EXTEND_RIGHT_CIP(ptr, start, length)                                   \

        for (i = start; i < (start) + (length); i += 4)                        \

            if (!IS_INTRA(i, -1))                                              \

                AV_WN4P(&ptr[i], a);                                           \

            else                                                               \

                a = PIXEL_SPLAT_X4(ptr[i+3])

#define EXTEND_LEFT_CIP(ptr, start, length) \

        for (i = start; i > (start) - (length); i--) \

            if (!IS_INTRA(i - 1, -1)) \

                ptr[i - 1] = ptr[i]

#define EXTEND_UP_CIP(ptr, start, length)                                      \

        for (i = (start); i > (start) - (length); i -= 4)                      \

            if (!IS_INTRA(-1, i - 3))                                          \

                AV_WN4P(&ptr[i - 3], a);                                       \

            else                                                               \

                a = PIXEL_SPLAT_X4(ptr[i - 3])

#define EXTEND_DOWN_CIP(ptr, start, length)                                    \

        for (i = start; i < (start) + (length); i += 4)                        \

            if (!IS_INTRA(-1, i))                                              \

                AV_WN4P(&ptr[i], a);                                           \

            else                                                               \

                a = PIXEL_SPLAT_X4(ptr[i + 3])



    HEVCLocalContext *lc = s->HEVClc;

    int i;

    int hshift = s->ps.sps->hshift[c_idx];

    int vshift = s->ps.sps->vshift[c_idx];

    int size = (1 << log2_size);

    int size_in_luma_h = size << hshift;

    int size_in_tbs_h  = size_in_luma_h >> s->ps.sps->log2_min_tb_size;

    int size_in_luma_v = size << vshift;

    int size_in_tbs_v  = size_in_luma_v >> s->ps.sps->log2_min_tb_size;

    int x = x0 >> hshift;

    int y = y0 >> vshift;

    int x_tb = (x0 >> s->ps.sps->log2_min_tb_size) & s->ps.sps->tb_mask;

    int y_tb = (y0 >> s->ps.sps->log2_min_tb_size) & s->ps.sps->tb_mask;



    int cur_tb_addr = MIN_TB_ADDR_ZS(x_tb, y_tb);



    ptrdiff_t stride = s->frame->linesize[c_idx] / sizeof(pixel);

    pixel *src = (pixel*)s->frame->data[c_idx] + x + y * stride;



    int min_pu_width = s->ps.sps->min_pu_width;



    enum IntraPredMode mode = c_idx ? lc->tu.intra_pred_mode_c :

                              lc->tu.intra_pred_mode;

    pixel4 a;

    pixel  left_array[2 * MAX_TB_SIZE + 1];

    pixel  filtered_left_array[2 * MAX_TB_SIZE + 1];

    pixel  top_array[2 * MAX_TB_SIZE + 1];

    pixel  filtered_top_array[2 * MAX_TB_SIZE + 1];



    pixel  *left          = left_array + 1;

    pixel  *top           = top_array  + 1;

    pixel  *filtered_left = filtered_left_array + 1;

    pixel  *filtered_top  = filtered_top_array  + 1;

    int cand_bottom_left = lc->na.cand_bottom_left && cur_tb_addr > MIN_TB_ADDR_ZS( x_tb - 1, (y_tb + size_in_tbs_v) & s->ps.sps->tb_mask);

    int cand_left        = lc->na.cand_left;

    int cand_up_left     = lc->na.cand_up_left;

    int cand_up          = lc->na.cand_up;

    int cand_up_right    = lc->na.cand_up_right    && cur_tb_addr > MIN_TB_ADDR_ZS((x_tb + size_in_tbs_h) & s->ps.sps->tb_mask, y_tb - 1);



    int bottom_left_size = (FFMIN(y0 + 2 * size_in_luma_v, s->ps.sps->height) -

                           (y0 + size_in_luma_v)) >> vshift;

    int top_right_size   = (FFMIN(x0 + 2 * size_in_luma_h, s->ps.sps->width) -

                           (x0 + size_in_luma_h)) >> hshift;



    if (s->ps.pps->constrained_intra_pred_flag == 1) {

        int size_in_luma_pu_v = PU(size_in_luma_v);

        int size_in_luma_pu_h = PU(size_in_luma_h);

        int on_pu_edge_x    = !av_mod_uintp2(x0, s->ps.sps->log2_min_pu_size);

        int on_pu_edge_y    = !av_mod_uintp2(y0, s->ps.sps->log2_min_pu_size);

        if (!size_in_luma_pu_h)

            size_in_luma_pu_h++;

        if (cand_bottom_left == 1 && on_pu_edge_x) {

            int x_left_pu   = PU(x0 - 1);

            int y_bottom_pu = PU(y0 + size_in_luma_v);

            int max = FFMIN(size_in_luma_pu_v, s->ps.sps->min_pu_height - y_bottom_pu);

            cand_bottom_left = 0;

            for (i = 0; i < max; i += 2)

                cand_bottom_left |= (MVF(x_left_pu, y_bottom_pu + i).pred_flag == PF_INTRA);

        }

        if (cand_left == 1 && on_pu_edge_x) {

            int x_left_pu   = PU(x0 - 1);

            int y_left_pu   = PU(y0);

            int max = FFMIN(size_in_luma_pu_v, s->ps.sps->min_pu_height - y_left_pu);

            cand_left = 0;

            for (i = 0; i < max; i += 2)

                cand_left |= (MVF(x_left_pu, y_left_pu + i).pred_flag == PF_INTRA);

        }

        if (cand_up_left == 1) {

            int x_left_pu   = PU(x0 - 1);

            int y_top_pu    = PU(y0 - 1);

            cand_up_left = MVF(x_left_pu, y_top_pu).pred_flag == PF_INTRA;

        }

        if (cand_up == 1 && on_pu_edge_y) {

            int x_top_pu    = PU(x0);

            int y_top_pu    = PU(y0 - 1);

            int max = FFMIN(size_in_luma_pu_h, s->ps.sps->min_pu_width - x_top_pu);

            cand_up = 0;

            for (i = 0; i < max; i += 2)

                cand_up |= (MVF(x_top_pu + i, y_top_pu).pred_flag == PF_INTRA);

        }

        if (cand_up_right == 1 && on_pu_edge_y) {

            int y_top_pu    = PU(y0 - 1);

            int x_right_pu  = PU(x0 + size_in_luma_h);

            int max = FFMIN(size_in_luma_pu_h, s->ps.sps->min_pu_width - x_right_pu);

            cand_up_right = 0;

            for (i = 0; i < max; i += 2)

                cand_up_right |= (MVF(x_right_pu + i, y_top_pu).pred_flag == PF_INTRA);

        }

        memset(left, 128, 2 * MAX_TB_SIZE*sizeof(pixel));

        memset(top , 128, 2 * MAX_TB_SIZE*sizeof(pixel));

        top[-1] = 128;

    }

    if (cand_up_left) {

        left[-1] = POS(-1, -1);

        top[-1]  = left[-1];

    }

    if (cand_up)

        memcpy(top, src - stride, size * sizeof(pixel));

    if (cand_up_right) {

        memcpy(top + size, src - stride + size, size * sizeof(pixel));

        EXTEND(top + size + top_right_size, POS(size + top_right_size - 1, -1),

               size - top_right_size);

    }

    if (cand_left)

        for (i = 0; i < size; i++)

            left[i] = POS(-1, i);

    if (cand_bottom_left) {

        for (i = size; i < size + bottom_left_size; i++)

            left[i] = POS(-1, i);

        EXTEND(left + size + bottom_left_size, POS(-1, size + bottom_left_size - 1),

               size - bottom_left_size);

    }



    if (s->ps.pps->constrained_intra_pred_flag == 1) {

        if (cand_bottom_left || cand_left || cand_up_left || cand_up || cand_up_right) {

            int size_max_x = x0 + ((2 * size) << hshift) < s->ps.sps->width ?

                                    2 * size : (s->ps.sps->width - x0) >> hshift;

            int size_max_y = y0 + ((2 * size) << vshift) < s->ps.sps->height ?

                                    2 * size : (s->ps.sps->height - y0) >> vshift;

            int j = size + (cand_bottom_left? bottom_left_size: 0) -1;

            if (!cand_up_right) {

                size_max_x = x0 + ((size) << hshift) < s->ps.sps->width ?

                                                    size : (s->ps.sps->width - x0) >> hshift;

            }

            if (!cand_bottom_left) {

                size_max_y = y0 + (( size) << vshift) < s->ps.sps->height ?

                                                     size : (s->ps.sps->height - y0) >> vshift;

            }

            if (cand_bottom_left || cand_left || cand_up_left) {

                while (j > -1 && !IS_INTRA(-1, j))

                    j--;

                if (!IS_INTRA(-1, j)) {

                    j = 0;

                    while (j < size_max_x && !IS_INTRA(j, -1))

                        j++;

                    EXTEND_LEFT_CIP(top, j, j + 1);

                    left[-1] = top[-1];

                }

            } else {

                j = 0;

                while (j < size_max_x && !IS_INTRA(j, -1))

                    j++;

                if (j > 0)

                    if (x0 > 0) {

                        EXTEND_LEFT_CIP(top, j, j + 1);

                    } else {

                        EXTEND_LEFT_CIP(top, j, j);

                        top[-1] = top[0];

                    }

                left[-1] = top[-1];

            }

            left[-1] = top[-1];

            if (cand_bottom_left || cand_left) {

                a = PIXEL_SPLAT_X4(left[-1]);

                EXTEND_DOWN_CIP(left, 0, size_max_y);

            }

            if (!cand_left)

                EXTEND(left, left[-1], size);

            if (!cand_bottom_left)

                EXTEND(left + size, left[size - 1], size);

            if (x0 != 0 && y0 != 0) {

                a = PIXEL_SPLAT_X4(left[size_max_y - 1]);

                EXTEND_UP_CIP(left, size_max_y - 1, size_max_y);

                if (!IS_INTRA(-1, - 1))

                    left[-1] = left[0];

            } else if (x0 == 0) {

                EXTEND(left, 0, size_max_y);

            } else {

                a = PIXEL_SPLAT_X4(left[size_max_y - 1]);

                EXTEND_UP_CIP(left, size_max_y - 1, size_max_y);

            }

            top[-1] = left[-1];

            if (y0 != 0) {

                a = PIXEL_SPLAT_X4(left[-1]);

                EXTEND_RIGHT_CIP(top, 0, size_max_x);

            }

        }

    }

    // Infer the unavailable samples

    if (!cand_bottom_left) {

        if (cand_left) {

            EXTEND(left + size, left[size - 1], size);

        } else if (cand_up_left) {

            EXTEND(left, left[-1], 2 * size);

            cand_left = 1;

        } else if (cand_up) {

            left[-1] = top[0];

            EXTEND(left, left[-1], 2 * size);

            cand_up_left = 1;

            cand_left    = 1;

        } else if (cand_up_right) {

            EXTEND(top, top[size], size);

            left[-1] = top[size];

            EXTEND(left, left[-1], 2 * size);

            cand_up      = 1;

            cand_up_left = 1;

            cand_left    = 1;

        } else { // No samples available

            left[-1] = (1 << (BIT_DEPTH - 1));

            EXTEND(top,  left[-1], 2 * size);

            EXTEND(left, left[-1], 2 * size);

        }

    }



    if (!cand_left)

        EXTEND(left, left[size], size);

    if (!cand_up_left) {

        left[-1] = left[0];

    }

    if (!cand_up)

        EXTEND(top, left[-1], size);

    if (!cand_up_right)

        EXTEND(top + size, top[size - 1], size);



    top[-1] = left[-1];



    // Filtering process

    if (!s->ps.sps->intra_smoothing_disabled_flag && (c_idx == 0  || s->ps.sps->chroma_format_idc == 3)) {

        if (mode != INTRA_DC && size != 4){

            int intra_hor_ver_dist_thresh[] = { 7, 1, 0 };

            int min_dist_vert_hor = FFMIN(FFABS((int)(mode - 26U)),

                                          FFABS((int)(mode - 10U)));

            if (min_dist_vert_hor > intra_hor_ver_dist_thresh[log2_size - 3]) {

                int threshold = 1 << (BIT_DEPTH - 5);

                if (s->ps.sps->sps_strong_intra_smoothing_enable_flag && c_idx == 0 &&

                    log2_size == 5 &&

                    FFABS(top[-1]  + top[63]  - 2 * top[31])  < threshold &&

                    FFABS(left[-1] + left[63] - 2 * left[31]) < threshold) {

                    // We can't just overwrite values in top because it could be

                    // a pointer into src

                    filtered_top[-1] = top[-1];

                    filtered_top[63] = top[63];

                    for (i = 0; i < 63; i++)

                        filtered_top[i] = ((64 - (i + 1)) * top[-1] +

                                           (i + 1)  * top[63] + 32) >> 6;

                    for (i = 0; i < 63; i++)

                        left[i] = ((64 - (i + 1)) * left[-1] +

                                   (i + 1)  * left[63] + 32) >> 6;

                    top = filtered_top;

                } else {

                    filtered_left[2 * size - 1] = left[2 * size - 1];

                    filtered_top[2 * size - 1]  = top[2 * size - 1];

                    for (i = 2 * size - 2; i >= 0; i--)

                        filtered_left[i] = (left[i + 1] + 2 * left[i] +

                                            left[i - 1] + 2) >> 2;

                    filtered_top[-1]  =

                    filtered_left[-1] = (left[0] + 2 * left[-1] + top[0] + 2) >> 2;

                    for (i = 2 * size - 2; i >= 0; i--)

                        filtered_top[i] = (top[i + 1] + 2 * top[i] +

                                           top[i - 1] + 2) >> 2;

                    left = filtered_left;

                    top  = filtered_top;

                }

            }

        }

    }



    switch (mode) {

    case INTRA_PLANAR:

        s->hpc.pred_planar[log2_size - 2]((uint8_t *)src, (uint8_t *)top,

                                          (uint8_t *)left, stride);

        break;

    case INTRA_DC:

        s->hpc.pred_dc((uint8_t *)src, (uint8_t *)top,

                       (uint8_t *)left, stride, log2_size, c_idx);

        break;

    default:

        s->hpc.pred_angular[log2_size - 2]((uint8_t *)src, (uint8_t *)top,

                                           (uint8_t *)left, stride, c_idx,

                                           mode);

        break;

    }

}
