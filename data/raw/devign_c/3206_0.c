static void RENAME(yuyvtoyuv420)(uint8_t *ydst, uint8_t *udst, uint8_t *vdst, const uint8_t *src,

                                      long width, long height,

                                      long lumStride, long chromStride, long srcStride)

{

    long y;

    const long chromWidth= -((-width)>>1);



    for (y=0; y<height; y++) {

        RENAME(extract_even)(src, ydst, width);

        if(y&1) {

            RENAME(extract_odd2avg)(src-srcStride, src, udst, vdst, chromWidth);

            udst+= chromStride;

            vdst+= chromStride;

        }



        src += srcStride;

        ydst+= lumStride;

    }

#if COMPILE_TEMPLATE_MMX

    __asm__(

            EMMS"       \n\t"

            SFENCE"     \n\t"

            ::: "memory"

        );

#endif

}
