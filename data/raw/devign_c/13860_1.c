static int dct_quantize_refine(MpegEncContext *s, //FIXME breaks denoise?

                        DCTELEM *block, int16_t *weight, DCTELEM *orig,

                        int n, int qscale){

    int16_t rem[64];

    DCTELEM d1[64];

    const int *qmat;

    const uint8_t *scantable= s->intra_scantable.scantable;

    const uint8_t *perm_scantable= s->intra_scantable.permutated;

//    unsigned int threshold1, threshold2;

//    int bias=0;

    int run_tab[65];

    int prev_run=0;

    int prev_level=0;

    int qmul, qadd, start_i, last_non_zero, i, dc;

    uint8_t * length;

    uint8_t * last_length;

    int lambda;

    int rle_index, run, q, sum;

#ifdef REFINE_STATS

static int count=0;

static int after_last=0;

static int to_zero=0;

static int from_zero=0;

static int raise=0;

static int lower=0;

static int messed_sign=0;

#endif



    if(basis[0][0] == 0)

        build_basis(s->dsp.idct_permutation);

    

    qmul= qscale*2;

    qadd= (qscale-1)|1;

    if (s->mb_intra) {

        if (!s->h263_aic) {

            if (n < 4)

                q = s->y_dc_scale;

            else

                q = s->c_dc_scale;

        } else{

            /* For AIC we skip quant/dequant of INTRADC */

            q = 1;

            qadd=0;

        }

        q <<= RECON_SHIFT-3;

        /* note: block[0] is assumed to be positive */

        dc= block[0]*q;

//        block[0] = (block[0] + (q >> 1)) / q;

        start_i = 1;

        qmat = s->q_intra_matrix[qscale];

//        if(s->mpeg_quant || s->out_format == FMT_MPEG1)

//            bias= 1<<(QMAT_SHIFT-1);

        length     = s->intra_ac_vlc_length;

        last_length= s->intra_ac_vlc_last_length;

    } else {

        dc= 0;

        start_i = 0;

        qmat = s->q_inter_matrix[qscale];

        length     = s->inter_ac_vlc_length;

        last_length= s->inter_ac_vlc_last_length;

    }

    last_non_zero = s->block_last_index[n];



#ifdef REFINE_STATS

{START_TIMER

#endif

    dc += (1<<(RECON_SHIFT-1));

    for(i=0; i<64; i++){

        rem[i]= dc - (orig[i]<<RECON_SHIFT); //FIXME  use orig dirrectly insteadof copying to rem[]

    }

#ifdef REFINE_STATS

STOP_TIMER("memset rem[]")}

#endif

    sum=0;

    for(i=0; i<64; i++){

        int one= 36;

        int qns=4;

        int w;



        w= ABS(weight[i]) + qns*one;

        w= 15 + (48*qns*one + w/2)/w; // 16 .. 63



        weight[i] = w;

//        w=weight[i] = (63*qns + (w/2)) / w;

         

        assert(w>0);

        assert(w<(1<<6));

        sum += w*w;

    }

    lambda= sum*(uint64_t)s->lambda2 >> (FF_LAMBDA_SHIFT - 6 + 6 + 6 + 6);

#ifdef REFINE_STATS

{START_TIMER

#endif

    run=0;

    rle_index=0;

    for(i=start_i; i<=last_non_zero; i++){

        int j= perm_scantable[i];

        const int level= block[j];

        int coeff;

        

        if(level){

            if(level<0) coeff= qmul*level - qadd;

            else        coeff= qmul*level + qadd;

            run_tab[rle_index++]=run;

            run=0;



            s->dsp.add_8x8basis(rem, basis[j], coeff);

        }else{

            run++;

        }

    }

#ifdef REFINE_STATS

if(last_non_zero>0){

STOP_TIMER("init rem[]")

}

}



{START_TIMER

#endif

    for(;;){

        int best_score=s->dsp.try_8x8basis(rem, weight, basis[0], 0);

        int best_coeff=0;

        int best_change=0;

        int run2, best_unquant_change=0, analyze_gradient;

#ifdef REFINE_STATS

{START_TIMER

#endif

        analyze_gradient = last_non_zero > 2 || s->avctx->quantizer_noise_shaping >= 3;



        if(analyze_gradient){

#ifdef REFINE_STATS

{START_TIMER

#endif

            for(i=0; i<64; i++){

                int w= weight[i];

            

                d1[i] = (rem[i]*w*w + (1<<(RECON_SHIFT+12-1)))>>(RECON_SHIFT+12);

            }

#ifdef REFINE_STATS

STOP_TIMER("rem*w*w")}

{START_TIMER

#endif

            s->dsp.fdct(d1);

#ifdef REFINE_STATS

STOP_TIMER("dct")}

#endif

        }



        if(start_i){

            const int level= block[0];

            int change, old_coeff;



            assert(s->mb_intra);

            

            old_coeff= q*level;

            

            for(change=-1; change<=1; change+=2){

                int new_level= level + change;

                int score, new_coeff;

                

                new_coeff= q*new_level;

                if(new_coeff >= 2048 || new_coeff < 0)

                    continue;



                score= s->dsp.try_8x8basis(rem, weight, basis[0], new_coeff - old_coeff);

                if(score<best_score){

                    best_score= score;

                    best_coeff= 0;

                    best_change= change;

                    best_unquant_change= new_coeff - old_coeff;

                }

            }

        }

        

        run=0;

        rle_index=0;

        run2= run_tab[rle_index++];

        prev_level=0;

        prev_run=0;



        for(i=start_i; i<64; i++){

            int j= perm_scantable[i];

            const int level= block[j];

            int change, old_coeff;



            if(s->avctx->quantizer_noise_shaping < 3 && i > last_non_zero + 1)

                break;



            if(level){

                if(level<0) old_coeff= qmul*level - qadd;

                else        old_coeff= qmul*level + qadd;

                run2= run_tab[rle_index++]; //FIXME ! maybe after last

            }else{

                old_coeff=0;

                run2--;

                assert(run2>=0 || i >= last_non_zero );

            }

            

            for(change=-1; change<=1; change+=2){

                int new_level= level + change;

                int score, new_coeff, unquant_change;

                

                score=0;

                if(s->avctx->quantizer_noise_shaping < 2 && ABS(new_level) > ABS(level))

                   continue;



                if(new_level){

                    if(new_level<0) new_coeff= qmul*new_level - qadd;

                    else            new_coeff= qmul*new_level + qadd;

                    if(new_coeff >= 2048 || new_coeff <= -2048)

                        continue;

                    //FIXME check for overflow

                    

                    if(level){

                        if(level < 63 && level > -63){

                            if(i < last_non_zero)

                                score +=   length[UNI_AC_ENC_INDEX(run, new_level+64)]

                                         - length[UNI_AC_ENC_INDEX(run, level+64)];

                            else

                                score +=   last_length[UNI_AC_ENC_INDEX(run, new_level+64)]

                                         - last_length[UNI_AC_ENC_INDEX(run, level+64)];

                        }

                    }else{

                        assert(ABS(new_level)==1);

                        

                        if(analyze_gradient){

                            int g= d1[ scantable[i] ];

                            if(g && (g^new_level) >= 0)

                                continue;

                        }



                        if(i < last_non_zero){

                            int next_i= i + run2 + 1;

                            int next_level= block[ perm_scantable[next_i] ] + 64;

                            

                            if(next_level&(~127))

                                next_level= 0;



                            if(next_i < last_non_zero)

                                score +=   length[UNI_AC_ENC_INDEX(run, 65)]

                                         + length[UNI_AC_ENC_INDEX(run2, next_level)]

                                         - length[UNI_AC_ENC_INDEX(run + run2 + 1, next_level)];

                            else

                                score +=  length[UNI_AC_ENC_INDEX(run, 65)]

                                        + last_length[UNI_AC_ENC_INDEX(run2, next_level)]

                                        - last_length[UNI_AC_ENC_INDEX(run + run2 + 1, next_level)];

                        }else{

                            score += last_length[UNI_AC_ENC_INDEX(run, 65)];

                            if(prev_level){

                                score +=  length[UNI_AC_ENC_INDEX(prev_run, prev_level)]

                                        - last_length[UNI_AC_ENC_INDEX(prev_run, prev_level)];

                            }

                        }

                    }

                }else{

                    new_coeff=0;

                    assert(ABS(level)==1);



                    if(i < last_non_zero){

                        int next_i= i + run2 + 1;

                        int next_level= block[ perm_scantable[next_i] ] + 64;

                            

                        if(next_level&(~127))

                            next_level= 0;



                        if(next_i < last_non_zero)

                            score +=   length[UNI_AC_ENC_INDEX(run + run2 + 1, next_level)]

                                     - length[UNI_AC_ENC_INDEX(run2, next_level)]

                                     - length[UNI_AC_ENC_INDEX(run, 65)];

                        else

                            score +=   last_length[UNI_AC_ENC_INDEX(run + run2 + 1, next_level)]

                                     - last_length[UNI_AC_ENC_INDEX(run2, next_level)]

                                     - length[UNI_AC_ENC_INDEX(run, 65)];

                    }else{

                        score += -last_length[UNI_AC_ENC_INDEX(run, 65)];

                        if(prev_level){

                            score +=  last_length[UNI_AC_ENC_INDEX(prev_run, prev_level)]

                                    - length[UNI_AC_ENC_INDEX(prev_run, prev_level)];

                        }

                    }

                }

                

                score *= lambda;



                unquant_change= new_coeff - old_coeff;

                assert((score < 100*lambda && score > -100*lambda) || lambda==0);

                

                score+= s->dsp.try_8x8basis(rem, weight, basis[j], unquant_change);

                if(score<best_score){

                    best_score= score;

                    best_coeff= i;

                    best_change= change;

                    best_unquant_change= unquant_change;

                }

            }

            if(level){

                prev_level= level + 64;

                if(prev_level&(~127))

                    prev_level= 0;

                prev_run= run;

                run=0;

            }else{

                run++;

            }

        }

#ifdef REFINE_STATS

STOP_TIMER("iterative step")}

#endif



        if(best_change){

            int j= perm_scantable[ best_coeff ];

            

            block[j] += best_change;

            

            if(best_coeff > last_non_zero){

                last_non_zero= best_coeff;

                assert(block[j]);

#ifdef REFINE_STATS

after_last++;

#endif

            }else{

#ifdef REFINE_STATS

if(block[j]){

    if(block[j] - best_change){

        if(ABS(block[j]) > ABS(block[j] - best_change)){

            raise++;

        }else{

            lower++;

        }

    }else{

        from_zero++;

    }

}else{

    to_zero++;

}

#endif

                for(; last_non_zero>=start_i; last_non_zero--){

                    if(block[perm_scantable[last_non_zero]])

                        break;

                }

            }

#ifdef REFINE_STATS

count++;

if(256*256*256*64 % count == 0){

    printf("after_last:%d to_zero:%d from_zero:%d raise:%d lower:%d sign:%d xyp:%d/%d/%d\n", after_last, to_zero, from_zero, raise, lower, messed_sign, s->mb_x, s->mb_y, s->picture_number);

}

#endif

            run=0;

            rle_index=0;

            for(i=start_i; i<=last_non_zero; i++){

                int j= perm_scantable[i];

                const int level= block[j];

        

                 if(level){

                     run_tab[rle_index++]=run;

                     run=0;

                 }else{

                     run++;

                 }

            }

            

            s->dsp.add_8x8basis(rem, basis[j], best_unquant_change);

        }else{

            break;

        }

    }

#ifdef REFINE_STATS

if(last_non_zero>0){

STOP_TIMER("iterative search")

}

}

#endif



    return last_non_zero;

}
