static void FUNC(hevc_h_loop_filter_luma)(uint8_t *pix, ptrdiff_t stride,

                                          int *beta, int *tc, uint8_t *no_p,

                                          uint8_t *no_q)

{

    FUNC(hevc_loop_filter_luma)(pix, stride, sizeof(pixel),

                                beta, tc, no_p, no_q);

}
