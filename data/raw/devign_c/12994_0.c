static void h263_v_loop_filter_mmx(uint8_t *src, int stride, int qscale)

{

    if (CONFIG_H263_DECODER || CONFIG_H263_ENCODER) {

        const int strength = ff_h263_loop_filter_strength[qscale];



        __asm__ volatile (

            H263_LOOP_FILTER



            "movq %%mm3, %1             \n\t"

            "movq %%mm4, %2             \n\t"

            "movq %%mm5, %0             \n\t"

            "movq %%mm6, %3             \n\t"

            : "+m"(*(uint64_t*)(src - 2 * stride)),

              "+m"(*(uint64_t*)(src - 1 * stride)),

              "+m"(*(uint64_t*)(src + 0 * stride)),

              "+m"(*(uint64_t*)(src + 1 * stride))

            : "g"(2 * strength), "m"(ff_pb_FC)

            );

    }

}
