void ff_sbrdsp_init_x86(SBRDSPContext *s)

{

    if (HAVE_YASM) {

        int mm_flags = av_get_cpu_flags();



        if (mm_flags & AV_CPU_FLAG_SSE) {

            s->sum_square = ff_sbr_sum_square_sse;

            s->hf_g_filt = ff_sbr_hf_g_filt_sse;

        }

    }

}
