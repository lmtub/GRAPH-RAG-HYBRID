static double get_qscale(MpegEncContext *s, RateControlEntry *rce, double rate_factor, int frame_num){

    RateControlContext *rcc= &s->rc_context;

    AVCodecContext *a= s->avctx;

    double q, bits;

    const int pict_type= rce->new_pict_type;

    const double mb_num= s->mb_num;

    int i;



    double const_values[]={

        M_PI,

        M_E,

        rce->i_tex_bits*rce->qscale,

        rce->p_tex_bits*rce->qscale,

        (rce->i_tex_bits + rce->p_tex_bits)*(double)rce->qscale,

        rce->mv_bits/mb_num,

        rce->pict_type == B_TYPE ? (rce->f_code + rce->b_code)*0.5 : rce->f_code,

        rce->i_count/mb_num,

        rce->mc_mb_var_sum/mb_num,

        rce->mb_var_sum/mb_num,

        rce->pict_type == I_TYPE,

        rce->pict_type == P_TYPE,

        rce->pict_type == B_TYPE,

        rcc->qscale_sum[pict_type] / (double)rcc->frame_count[pict_type],

        a->qcompress,

/*        rcc->last_qscale_for[I_TYPE],

        rcc->last_qscale_for[P_TYPE],

        rcc->last_qscale_for[B_TYPE],

        rcc->next_non_b_qscale,*/

        rcc->i_cplx_sum[I_TYPE] / (double)rcc->frame_count[I_TYPE],

        rcc->i_cplx_sum[P_TYPE] / (double)rcc->frame_count[P_TYPE],

        rcc->p_cplx_sum[P_TYPE] / (double)rcc->frame_count[P_TYPE],

        rcc->p_cplx_sum[B_TYPE] / (double)rcc->frame_count[B_TYPE],

        (rcc->i_cplx_sum[pict_type] + rcc->p_cplx_sum[pict_type]) / (double)rcc->frame_count[pict_type],

        0

    };



    bits= ff_parse_eval(rcc->rc_eq_eval, const_values, rce);

    if (isnan(bits)) {

        av_log(s->avctx, AV_LOG_ERROR, "Error evaluating rc_eq \"%s\"\n", s->avctx->rc_eq);

        return -1;

    }



    rcc->pass1_rc_eq_output_sum+= bits;

    bits*=rate_factor;

    if(bits<0.0) bits=0.0;

    bits+= 1.0; //avoid 1/0 issues



    /* user override */

    for(i=0; i<s->avctx->rc_override_count; i++){

        RcOverride *rco= s->avctx->rc_override;

        if(rco[i].start_frame > frame_num) continue;

        if(rco[i].end_frame   < frame_num) continue;



        if(rco[i].qscale)

            bits= qp2bits(rce, rco[i].qscale); //FIXME move at end to really force it?

        else

            bits*= rco[i].quality_factor;

    }



    q= bits2qp(rce, bits);



    /* I/B difference */

    if     (pict_type==I_TYPE && s->avctx->i_quant_factor<0.0)

        q= -q*s->avctx->i_quant_factor + s->avctx->i_quant_offset;

    else if(pict_type==B_TYPE && s->avctx->b_quant_factor<0.0)

        q= -q*s->avctx->b_quant_factor + s->avctx->b_quant_offset;




    return q;

}