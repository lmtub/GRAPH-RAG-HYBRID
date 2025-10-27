static int16_t long_term_filter(DSPContext *dsp, int pitch_delay_int,

                                const int16_t* residual, int16_t *residual_filt,

                                int subframe_size)

{

    int i, k, tmp, tmp2;

    int sum;

    int L_temp0;

    int L_temp1;

    int64_t L64_temp0;

    int64_t L64_temp1;

    int16_t shift;

    int corr_int_num, corr_int_den;



    int ener;

    int16_t sh_ener;



    int16_t gain_num,gain_den; //selected signal's gain numerator and denominator

    int16_t sh_gain_num, sh_gain_den;

    int gain_num_square;



    int16_t gain_long_num,gain_long_den; //filtered through long interpolation filter signal's gain numerator and denominator

    int16_t sh_gain_long_num, sh_gain_long_den;



    int16_t best_delay_int, best_delay_frac;



    int16_t delayed_signal_offset;

    int lt_filt_factor_a, lt_filt_factor_b;



    int16_t * selected_signal;

    const int16_t * selected_signal_const; //Necessary to avoid compiler warning



    int16_t sig_scaled[SUBFRAME_SIZE + RES_PREV_DATA_SIZE];

    int16_t delayed_signal[ANALYZED_FRAC_DELAYS][SUBFRAME_SIZE+1];

    int corr_den[ANALYZED_FRAC_DELAYS][2];



    tmp = 0;

    for(i=0; i<subframe_size + RES_PREV_DATA_SIZE; i++)

        tmp |= FFABS(residual[i]);



    if(!tmp)

        shift = 3;

    else

        shift = av_log2(tmp) - 11;



    if (shift > 0)

        for (i = 0; i < subframe_size + RES_PREV_DATA_SIZE; i++)

            sig_scaled[i] = residual[i] >> shift;

    else

        for (i = 0; i < subframe_size + RES_PREV_DATA_SIZE; i++)

            sig_scaled[i] = residual[i] << -shift;



    /* Start of best delay searching code */

    gain_num = 0;



    ener = dsp->scalarproduct_int16(sig_scaled + RES_PREV_DATA_SIZE,

                                    sig_scaled + RES_PREV_DATA_SIZE,

                                    subframe_size);

    if (ener) {

        sh_ener = FFMAX(av_log2(ener) - 14, 0);

        ener >>= sh_ener;

        /* Search for best pitch delay.



                       sum{ r(n) * r(k,n) ] }^2

           R'(k)^2 := -------------------------

                       sum{ r(k,n) * r(k,n) }





           R(T)    :=  sum{ r(n) * r(n-T) ] }





           where

           r(n-T) is integer delayed signal with delay T

           r(k,n) is non-integer delayed signal with integer delay best_delay

           and fractional delay k */



        /* Find integer delay best_delay which maximizes correlation R(T).



           This is also equals to numerator of R'(0),

           since the fine search (second step) is done with 1/8

           precision around best_delay. */

        corr_int_num = 0;

        best_delay_int = pitch_delay_int - 1;

        for (i = pitch_delay_int - 1; i <= pitch_delay_int + 1; i++) {

            sum = dsp->scalarproduct_int16(sig_scaled + RES_PREV_DATA_SIZE,

                                           sig_scaled + RES_PREV_DATA_SIZE - i,

                                           subframe_size);

            if (sum > corr_int_num) {

                corr_int_num = sum;

                best_delay_int = i;

            }

        }

        if (corr_int_num) {

            /* Compute denominator of pseudo-normalized correlation R'(0). */

            corr_int_den = dsp->scalarproduct_int16(sig_scaled - best_delay_int + RES_PREV_DATA_SIZE,

                                                    sig_scaled - best_delay_int + RES_PREV_DATA_SIZE,

                                                    subframe_size);



            /* Compute signals with non-integer delay k (with 1/8 precision),

               where k is in [0;6] range.

               Entire delay is qual to best_delay+(k+1)/8

               This is archieved by applying an interpolation filter of

               legth 33 to source signal. */

            for (k = 0; k < ANALYZED_FRAC_DELAYS; k++) {

                ff_acelp_interpolate(&delayed_signal[k][0],

                                     &sig_scaled[RES_PREV_DATA_SIZE - best_delay_int],

                                     ff_g729_interp_filt_short,

                                     ANALYZED_FRAC_DELAYS+1,

                                     8 - k - 1,

                                     SHORT_INT_FILT_LEN,

                                     subframe_size + 1);

            }



            /* Compute denominator of pseudo-normalized correlation R'(k).



                 corr_den[k][0] is square root of R'(k) denominator, for int(T) == int(T0)

                 corr_den[k][1] is square root of R'(k) denominator, for int(T) == int(T0)+1



              Also compute maximum value of above denominators over all k. */

            tmp = corr_int_den;

            for (k = 0; k < ANALYZED_FRAC_DELAYS; k++) {

                sum = dsp->scalarproduct_int16(&delayed_signal[k][1],

                                               &delayed_signal[k][1],

                                               subframe_size - 1);

                corr_den[k][0] = sum + delayed_signal[k][0            ] * delayed_signal[k][0            ];

                corr_den[k][1] = sum + delayed_signal[k][subframe_size] * delayed_signal[k][subframe_size];



                tmp = FFMAX3(tmp, corr_den[k][0], corr_den[k][1]);

            }



            sh_gain_den = av_log2(tmp) - 14;

            if (sh_gain_den >= 0) {



                sh_gain_num =  FFMAX(sh_gain_den, sh_ener);

                /* Loop through all k and find delay that maximizes

                   R'(k) correlation.

                   Search is done in [int(T0)-1; intT(0)+1] range

                   with 1/8 precision. */

                delayed_signal_offset = 1;

                best_delay_frac = 0;

                gain_den = corr_int_den >> sh_gain_den;

                gain_num = corr_int_num >> sh_gain_num;

                gain_num_square = gain_num * gain_num;

                for (k = 0; k < ANALYZED_FRAC_DELAYS; k++) {

                    for (i = 0; i < 2; i++) {

                        int16_t gain_num_short, gain_den_short;

                        int gain_num_short_square;

                        /* Compute numerator of pseudo-normalized

                           correlation R'(k). */

                        sum = dsp->scalarproduct_int16(&delayed_signal[k][i],

                                                       sig_scaled + RES_PREV_DATA_SIZE,

                                                       subframe_size);

                        gain_num_short = FFMAX(sum >> sh_gain_num, 0);



                        /*

                                      gain_num_short_square                gain_num_square

                           R'(T)^2 = -----------------------, max R'(T)^2= --------------

                                           den                                 gain_den

                        */

                        gain_num_short_square = gain_num_short * gain_num_short;

                        gain_den_short = corr_den[k][i] >> sh_gain_den;



                        tmp = MULL(gain_num_short_square, gain_den, FRAC_BITS);

                        tmp2 = MULL(gain_num_square, gain_den_short, FRAC_BITS);



                        // R'(T)^2 > max R'(T)^2

                        if (tmp > tmp2) {

                            gain_num = gain_num_short;

                            gain_den = gain_den_short;

                            gain_num_square = gain_num_short_square;

                            delayed_signal_offset = i;

                            best_delay_frac = k + 1;

                        }

                    }

                }



                /*

                       R'(T)^2

                  2 * --------- < 1

                        R(0)

                */

                L64_temp0 =  (int64_t)gain_num_square  << ((sh_gain_num << 1) + 1);

                L64_temp1 = ((int64_t)gain_den * ener) << (sh_gain_den + sh_ener);

                if (L64_temp0 < L64_temp1)

                    gain_num = 0;

            } // if(sh_gain_den >= 0)

        } // if(corr_int_num)

    } // if(ener)

    /* End of best delay searching code  */



    if (!gain_num) {

        memcpy(residual_filt, residual + RES_PREV_DATA_SIZE, subframe_size * sizeof(int16_t));



        /* Long-term prediction gain is less than 3dB. Long-term postfilter is disabled. */

        return 0;

    }

    if (best_delay_frac) {

        /* Recompute delayed signal with an interpolation filter of length 129. */

        ff_acelp_interpolate(residual_filt,

                             &sig_scaled[RES_PREV_DATA_SIZE - best_delay_int + delayed_signal_offset],

                             ff_g729_interp_filt_long,

                             ANALYZED_FRAC_DELAYS + 1,

                             8 - best_delay_frac,

                             LONG_INT_FILT_LEN,

                             subframe_size + 1);

        /* Compute R'(k) correlation's numerator. */

        sum = dsp->scalarproduct_int16(residual_filt,

                                       sig_scaled + RES_PREV_DATA_SIZE,

                                       subframe_size);



        if (sum < 0) {

            gain_long_num = 0;

            sh_gain_long_num = 0;

        } else {

            tmp = FFMAX(av_log2(sum) - 14, 0);

            sum >>= tmp;

            gain_long_num = sum;

            sh_gain_long_num = tmp;

        }



        /* Compute R'(k) correlation's denominator. */

        sum = dsp->scalarproduct_int16(residual_filt, residual_filt, subframe_size);



        tmp = FFMAX(av_log2(sum) - 14, 0);

        sum >>= tmp;

        gain_long_den = sum;

        sh_gain_long_den = tmp;



        /* Select between original and delayed signal.

           Delayed signal will be selected if it increases R'(k)

           correlation. */

        L_temp0 = gain_num * gain_num;

        L_temp0 = MULL(L_temp0, gain_long_den, FRAC_BITS);



        L_temp1 = gain_long_num * gain_long_num;

        L_temp1 = MULL(L_temp1, gain_den, FRAC_BITS);



        tmp = ((sh_gain_long_num - sh_gain_num) << 1) - (sh_gain_long_den - sh_gain_den);

        if (tmp > 0)

            L_temp0 >>= tmp;

        else

            L_temp1 >>= -tmp;



        /* Check if longer filter increases the values of R'(k). */

        if (L_temp1 > L_temp0) {

            /* Select long filter. */

            selected_signal = residual_filt;

            gain_num = gain_long_num;

            gain_den = gain_long_den;

            sh_gain_num = sh_gain_long_num;

            sh_gain_den = sh_gain_long_den;

        } else

            /* Select short filter. */

            selected_signal = &delayed_signal[best_delay_frac-1][delayed_signal_offset];



        /* Rescale selected signal to original value. */

        if (shift > 0)

            for (i = 0; i < subframe_size; i++)

                selected_signal[i] <<= shift;

        else

            for (i = 0; i < subframe_size; i++)

                selected_signal[i] >>= -shift;



        /* necessary to avoid compiler warning */

        selected_signal_const = selected_signal;

    } // if(best_delay_frac)

    else

        selected_signal_const = residual + RES_PREV_DATA_SIZE - (best_delay_int + 1 - delayed_signal_offset);

#ifdef G729_BITEXACT

    tmp = sh_gain_num - sh_gain_den;

    if (tmp > 0)

        gain_den >>= tmp;

    else

        gain_num >>= -tmp;



    if (gain_num > gain_den)

        lt_filt_factor_a = MIN_LT_FILT_FACTOR_A;

    else {

        gain_num >>= 2;

        gain_den >>= 1;

        lt_filt_factor_a = (gain_den << 15) / (gain_den + gain_num);

    }

#else

    L64_temp0 = ((int64_t)gain_num) << (sh_gain_num - 1);

    L64_temp1 = ((int64_t)gain_den) << sh_gain_den;

    lt_filt_factor_a = FFMAX((L64_temp1 << 15) / (L64_temp1 + L64_temp0), MIN_LT_FILT_FACTOR_A);

#endif



    /* Filter through selected filter. */

    lt_filt_factor_b = 32767 - lt_filt_factor_a + 1;



    ff_acelp_weighted_vector_sum(residual_filt, residual + RES_PREV_DATA_SIZE,

                                 selected_signal_const,

                                 lt_filt_factor_a, lt_filt_factor_b,

                                 1<<14, 15, subframe_size);



    // Long-term prediction gain is larger than 3dB.

    return 1;

}
