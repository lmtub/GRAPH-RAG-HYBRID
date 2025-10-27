static void FUNC(hevc_v_loop_filter_luma)(uint8_t *pix, ptrdiff_t stride,

                                          int *beta, int *tc, uint8_t *no_p,

                                          uint8_t *no_q)

{

    FUNC(hevc_loop_filter_luma)(pix, sizeof(pixel), stride,

                                beta, tc, no_p, no_q);

}
