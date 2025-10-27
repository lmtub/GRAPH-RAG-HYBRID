static void h264_loop_filter_strength_mmx2( int16_t bS[2][4][4], uint8_t nnz[40], int8_t ref[2][40], int16_t mv[2][40][2],

                                            int bidir, int edges, int step, int mask_mv0, int mask_mv1, int field ) {

    __asm__ volatile(

        "movq %0, %%mm7 \n"

        "movq %1, %%mm6 \n"

        ::"m"(ff_pb_1), "m"(ff_pb_3)

    );

    if(field)

        __asm__ volatile(

            "movq %0, %%mm6 \n"

            ::"m"(ff_pb_3_1)

        );

    __asm__ volatile(

        "movq  %%mm6, %%mm5 \n"

        "paddb %%mm5, %%mm5 \n"

    :);



    // could do a special case for dir==0 && edges==1, but it only reduces the

    // average filter time by 1.2%

    step  <<= 3;

    edges <<= 3;

    h264_loop_filter_strength_iteration_mmx2(bS, nnz, ref, mv, bidir, edges, step, mask_mv1, 1, -8,  0);

    h264_loop_filter_strength_iteration_mmx2(bS, nnz, ref, mv, bidir,    32,    8, mask_mv0, 0, -1, -1);



    __asm__ volatile(

        "movq   (%0), %%mm0 \n\t"

        "movq  8(%0), %%mm1 \n\t"

        "movq 16(%0), %%mm2 \n\t"

        "movq 24(%0), %%mm3 \n\t"

        TRANSPOSE4(%%mm0, %%mm1, %%mm2, %%mm3, %%mm4)

        "movq %%mm0,   (%0) \n\t"

        "movq %%mm3,  8(%0) \n\t"

        "movq %%mm4, 16(%0) \n\t"

        "movq %%mm2, 24(%0) \n\t"

        ::"r"(bS[0])

        :"memory"

    );

}
