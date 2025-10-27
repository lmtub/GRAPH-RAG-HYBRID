static void rv40_v_weak_loop_filter(uint8_t *src, const int stride,

                                    const int filter_p1, const int filter_q1,

                                    const int alpha, const int beta,

                                    const int lim_p0q0, const int lim_q1,

                                    const int lim_p1)

{

    rv40_weak_loop_filter(src, 1, stride, filter_p1, filter_q1,

                          alpha, beta, lim_p0q0, lim_q1, lim_p1);

}
