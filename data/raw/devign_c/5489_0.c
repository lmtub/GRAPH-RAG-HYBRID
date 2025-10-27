av_cold void ff_flacdsp_init_x86(FLACDSPContext *c, enum AVSampleFormat fmt,

                                 int bps)

{

#if HAVE_YASM

    int cpu_flags = av_get_cpu_flags();



    if (EXTERNAL_SSE4(cpu_flags)) {

        if (bps > 16 && CONFIG_FLAC_DECODER)

            c->lpc = ff_flac_lpc_32_sse4;

    }

    if (EXTERNAL_XOP(cpu_flags)) {

        if (bps > 16)

            c->lpc = ff_flac_lpc_32_xop;

    }

#endif

}
