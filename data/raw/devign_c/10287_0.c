static void float_to_int16_sse(int16_t *dst, const float *src, long len){

    int i;

    for(i=0; i<len; i+=4) {

        asm volatile(

            "cvtps2pi    %1, %%mm0 \n\t"

            "cvtps2pi    %2, %%mm1 \n\t"

            "packssdw %%mm1, %%mm0 \n\t"

            "movq     %%mm0, %0    \n\t"

            :"=m"(dst[i])

            :"m"(src[i]), "m"(src[i+2])

        );

    }

    asm volatile("emms");

}
