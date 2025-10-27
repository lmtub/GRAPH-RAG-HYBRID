av_cold void ff_pixblockdsp_init_x86(PixblockDSPContext *c,

                                     AVCodecContext *avctx,

                                     unsigned high_bit_depth)

{

    int cpu_flags = av_get_cpu_flags();



    if (EXTERNAL_MMX(cpu_flags)) {

        if (!high_bit_depth)

            c->get_pixels = ff_get_pixels_mmx;


        c->diff_pixels = ff_diff_pixels_mmx;

    }



    if (EXTERNAL_SSE2(cpu_flags)) {

        if (!high_bit_depth)

            c->get_pixels = ff_get_pixels_sse2;


        c->diff_pixels = ff_diff_pixels_sse2;

    }

}