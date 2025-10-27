static void RENAME(uyvytoyuv422)(uint8_t *ydst, uint8_t *udst, uint8_t *vdst, const uint8_t *src,

                                      long width, long height,

                                      long lumStride, long chromStride, long srcStride)

{

    long y;

    const long chromWidth= -((-width)>>1);



    for (y=0; y<height; y++) {

        RENAME(extract_even)(src+1, ydst, width);

        RENAME(extract_even2)(src, udst, vdst, chromWidth);



        src += srcStride;

        ydst+= lumStride;

        udst+= chromStride;

        vdst+= chromStride;

    }

#if COMPILE_TEMPLATE_MMX

    __asm__(

            EMMS"       \n\t"

            SFENCE"     \n\t"

            ::: "memory"

        );

#endif

}
