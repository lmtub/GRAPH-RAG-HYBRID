av_cold void ff_init_lls_x86(LLSModel *m)

{

    int cpu_flags = av_get_cpu_flags();

    if (EXTERNAL_SSE2(cpu_flags)) {

        m->update_lls = ff_update_lls_sse2;

        if (m->indep_count >= 4)

            m->evaluate_lls = ff_evaluate_lls_sse2;

    }

    if (EXTERNAL_AVX(cpu_flags)) {

        m->update_lls = ff_update_lls_avx;

    }

}
