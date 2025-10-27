static void ff_h264_idct8_add_mmx(uint8_t *dst, int16_t *block, int stride)

{

    int i;

    DECLARE_ALIGNED(8, int16_t, b2)[64];



    block[0] += 32;



    for(i=0; i<2; i++){

        DECLARE_ALIGNED(8, uint64_t, tmp);



        h264_idct8_1d(block+4*i);



        __asm__ volatile(

            "movq   %%mm7,    %0   \n\t"

            TRANSPOSE4( %%mm0, %%mm2, %%mm4, %%mm6, %%mm7 )

            "movq   %%mm0,  8(%1)  \n\t"

            "movq   %%mm6, 24(%1)  \n\t"

            "movq   %%mm7, 40(%1)  \n\t"

            "movq   %%mm4, 56(%1)  \n\t"

            "movq    %0,    %%mm7  \n\t"

            TRANSPOSE4( %%mm7, %%mm5, %%mm3, %%mm1, %%mm0 )

            "movq   %%mm7,   (%1)  \n\t"

            "movq   %%mm1, 16(%1)  \n\t"

            "movq   %%mm0, 32(%1)  \n\t"

            "movq   %%mm3, 48(%1)  \n\t"

            : "=m"(tmp)

            : "r"(b2+32*i)

            : "memory"

        );

    }



    for(i=0; i<2; i++){

        h264_idct8_1d(b2+4*i);



        __asm__ volatile(

            "psraw     $6, %%mm7  \n\t"

            "psraw     $6, %%mm6  \n\t"

            "psraw     $6, %%mm5  \n\t"

            "psraw     $6, %%mm4  \n\t"

            "psraw     $6, %%mm3  \n\t"

            "psraw     $6, %%mm2  \n\t"

            "psraw     $6, %%mm1  \n\t"

            "psraw     $6, %%mm0  \n\t"



            "movq   %%mm7,    (%0)  \n\t"

            "movq   %%mm5,  16(%0)  \n\t"

            "movq   %%mm3,  32(%0)  \n\t"

            "movq   %%mm1,  48(%0)  \n\t"

            "movq   %%mm0,  64(%0)  \n\t"

            "movq   %%mm2,  80(%0)  \n\t"

            "movq   %%mm4,  96(%0)  \n\t"

            "movq   %%mm6, 112(%0)  \n\t"

            :: "r"(b2+4*i)

            : "memory"

        );

    }



    ff_add_pixels_clamped_mmx(b2, dst, stride);

}
