static void hb_synthesis(AMRWBContext *ctx, int subframe, float *samples,

                         const float *exc, const float *isf, const float *isf_past)

{

    float hb_lpc[LP_ORDER_16k];

    enum Mode mode = ctx->fr_cur_mode;



    if (mode == MODE_6k60) {

        float e_isf[LP_ORDER_16k]; // ISF vector for extrapolation

        double e_isp[LP_ORDER_16k];



        ff_weighted_vector_sumf(e_isf, isf_past, isf, isfp_inter[subframe],

                                1.0 - isfp_inter[subframe], LP_ORDER);



        extrapolate_isf(e_isf, e_isf);



        e_isf[LP_ORDER_16k - 1] *= 2.0;

        ff_acelp_lsf2lspd(e_isp, e_isf, LP_ORDER_16k);

        ff_amrwb_lsp2lpc(e_isp, hb_lpc, LP_ORDER_16k);



        lpc_weighting(hb_lpc, hb_lpc, 0.9, LP_ORDER_16k);

    } else {

        lpc_weighting(hb_lpc, ctx->lp_coef[subframe], 0.6, LP_ORDER);

    }



    ff_celp_lp_synthesis_filterf(samples, hb_lpc, exc, AMRWB_SFR_SIZE_16k,

                                 (mode == MODE_6k60) ? LP_ORDER_16k : LP_ORDER);

}
