static void vp8_h_loop_filter_simple_c(uint8_t *dst, ptrdiff_t stride, int flim)

{

    int i;



    for (i = 0; i < 16; i++)

        if (simple_limit(dst + i * stride, 1, flim))

            filter_common(dst + i * stride, 1, 1);

}
