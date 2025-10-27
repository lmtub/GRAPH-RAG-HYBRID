static inline void transpose4x4(uint8_t *dst, uint8_t *src, x86_reg dst_stride, x86_reg src_stride){

    __asm__ volatile( //FIXME could save 1 instruction if done as 8x4 ...

        "movd  (%1), %%mm0              \n\t"

        "add   %3, %1                   \n\t"

        "movd  (%1), %%mm1              \n\t"

        "movd  (%1,%3,1), %%mm2         \n\t"

        "movd  (%1,%3,2), %%mm3         \n\t"

        "punpcklbw %%mm1, %%mm0         \n\t"

        "punpcklbw %%mm3, %%mm2         \n\t"

        "movq %%mm0, %%mm1              \n\t"

        "punpcklwd %%mm2, %%mm0         \n\t"

        "punpckhwd %%mm2, %%mm1         \n\t"

        "movd  %%mm0, (%0)              \n\t"

        "add   %2, %0                   \n\t"

        "punpckhdq %%mm0, %%mm0         \n\t"

        "movd  %%mm0, (%0)              \n\t"

        "movd  %%mm1, (%0,%2,1)         \n\t"

        "punpckhdq %%mm1, %%mm1         \n\t"

        "movd  %%mm1, (%0,%2,2)         \n\t"



        :  "+&r" (dst),

           "+&r" (src)

        :  "r" (dst_stride),

           "r" (src_stride)

        :  "memory"

    );

}
