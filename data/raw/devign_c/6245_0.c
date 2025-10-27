av_cold void ff_ps_ctx_init(PSContext *ps)

{

    ipdopd_reset(ps->ipd_hist, ps->opd_hist);

}
