av_cold void ff_blockdsp_init_x86(BlockDSPContext *c,

                                  AVCodecContext *avctx)

#else

av_cold void ff_blockdsp_init_x86(BlockDSPContext *c)

#endif /* FF_API_XVMC */

{

#if HAVE_INLINE_ASM

    int cpu_flags = av_get_cpu_flags();



    if (INLINE_MMX(cpu_flags)) {

        c->clear_block  = clear_block_mmx;

        c->clear_blocks = clear_blocks_mmx;

    }



#if FF_API_XVMC

FF_DISABLE_DEPRECATION_WARNINGS

    /* XvMCCreateBlocks() may not allocate 16-byte aligned blocks */

    if (CONFIG_MPEG_XVMC_DECODER && avctx->xvmc_acceleration > 1)

        return;

FF_ENABLE_DEPRECATION_WARNINGS

#endif /* FF_API_XVMC */



    if (INLINE_SSE(cpu_flags)) {

        c->clear_block  = clear_block_sse;

        c->clear_blocks = clear_blocks_sse;

    }

#endif /* HAVE_INLINE_ASM */

}
