int ff_lpc_calc_coefs(LPCContext *s,

                      const int32_t *samples, int blocksize, int min_order,

                      int max_order, int precision,

                      int32_t coefs[][MAX_LPC_ORDER], int *shift,

                      enum FFLPCType lpc_type, int lpc_passes,

                      int omethod, int max_shift, int zero_shift)

{

    double autoc[MAX_LPC_ORDER+1];

    double ref[MAX_LPC_ORDER];

    double lpc[MAX_LPC_ORDER][MAX_LPC_ORDER];

    int i, j, pass;

    int opt_order;



    av_assert2(max_order >= MIN_LPC_ORDER && max_order <= MAX_LPC_ORDER &&

           lpc_type > FF_LPC_TYPE_FIXED);



    /* reinit LPC context if parameters have changed */

    if (blocksize != s->blocksize || max_order != s->max_order ||

        lpc_type  != s->lpc_type) {

        ff_lpc_end(s);

        ff_lpc_init(s, blocksize, max_order, lpc_type);

    }



    if (lpc_type == FF_LPC_TYPE_LEVINSON) {

        s->lpc_apply_welch_window(samples, blocksize, s->windowed_samples);



        s->lpc_compute_autocorr(s->windowed_samples, blocksize, max_order, autoc);



        compute_lpc_coefs(autoc, max_order, &lpc[0][0], MAX_LPC_ORDER, 0, 1);



        for(i=0; i<max_order; i++)

            ref[i] = fabs(lpc[i][i]);

    } else if (lpc_type == FF_LPC_TYPE_CHOLESKY) {

        LLSModel m[2];

        double var[MAX_LPC_ORDER+1], av_uninit(weight);



        if(lpc_passes <= 0)

            lpc_passes = 2;



        for(pass=0; pass<lpc_passes; pass++){

            av_init_lls(&m[pass&1], max_order);



            weight=0;

            for(i=max_order; i<blocksize; i++){

                for(j=0; j<=max_order; j++)

                    var[j]= samples[i-j];



                if(pass){

                    double eval, inv, rinv;

                    eval= av_evaluate_lls(&m[(pass-1)&1], var+1, max_order-1);

                    eval= (512>>pass) + fabs(eval - var[0]);

                    inv = 1/eval;

                    rinv = sqrt(inv);

                    for(j=0; j<=max_order; j++)

                        var[j] *= rinv;

                    weight += inv;

                }else

                    weight++;



                av_update_lls(&m[pass&1], var, 1.0);

            }

            av_solve_lls(&m[pass&1], 0.001, 0);

        }



        for(i=0; i<max_order; i++){

            for(j=0; j<max_order; j++)

                lpc[i][j]=-m[(pass-1)&1].coeff[i][j];

            ref[i]= sqrt(m[(pass-1)&1].variance[i] / weight) * (blocksize - max_order) / 4000;

        }

        for(i=max_order-1; i>0; i--)

            ref[i] = ref[i-1] - ref[i];

    }

    opt_order = max_order;



    if(omethod == ORDER_METHOD_EST) {

        opt_order = estimate_best_order(ref, min_order, max_order);

        i = opt_order-1;

        quantize_lpc_coefs(lpc[i], i+1, precision, coefs[i], &shift[i], max_shift, zero_shift);

    } else {

        for(i=min_order-1; i<max_order; i++) {

            quantize_lpc_coefs(lpc[i], i+1, precision, coefs[i], &shift[i], max_shift, zero_shift);

        }

    }



    return opt_order;

}
