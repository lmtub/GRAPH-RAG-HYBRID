static void adaptive_quantization(MpegEncContext *s, double q){

    int i;

    const float lumi_masking= s->avctx->lumi_masking / (128.0*128.0);

    const float dark_masking= s->avctx->dark_masking / (128.0*128.0);

    const float temp_cplx_masking= s->avctx->temporal_cplx_masking;

    const float spatial_cplx_masking = s->avctx->spatial_cplx_masking;

    const float p_masking = s->avctx->p_masking;

    float bits_sum= 0.0;

    float cplx_sum= 0.0;

    float cplx_tab[s->mb_num];

    float bits_tab[s->mb_num];

    const int qmin= 2; //s->avctx->mb_qmin;

    const int qmax= 31; //s->avctx->mb_qmax;

    Picture * const pic= &s->current_picture;

    

    for(i=0; i<s->mb_num; i++){

        float temp_cplx= sqrt(pic->mc_mb_var[i]);

        float spat_cplx= sqrt(pic->mb_var[i]);

        const int lumi= pic->mb_mean[i];

        float bits, cplx, factor;

        

        if(spat_cplx < q/3) spat_cplx= q/3; //FIXME finetune

        if(temp_cplx < q/3) temp_cplx= q/3; //FIXME finetune

        

        if((s->mb_type[i]&MB_TYPE_INTRA)){//FIXME hq mode 

            cplx= spat_cplx;

            factor= 1.0 + p_masking;

        }else{

            cplx= temp_cplx;

            factor= pow(temp_cplx, - temp_cplx_masking);

        }

        factor*=pow(spat_cplx, - spatial_cplx_masking);



        if(lumi>127)

            factor*= (1.0 - (lumi-128)*(lumi-128)*lumi_masking);

        else

            factor*= (1.0 - (lumi-128)*(lumi-128)*dark_masking);

        

        if(factor<0.00001) factor= 0.00001;

        

        bits= cplx*factor;

        cplx_sum+= cplx;

        bits_sum+= bits;

        cplx_tab[i]= cplx;

        bits_tab[i]= bits;

    }



    /* handle qmin/qmax cliping */

    if(s->flags&CODEC_FLAG_NORMALIZE_AQP){

        for(i=0; i<s->mb_num; i++){

            float newq= q*cplx_tab[i]/bits_tab[i];

            newq*= bits_sum/cplx_sum;



            if     (newq > qmax){

                bits_sum -= bits_tab[i];

                cplx_sum -= cplx_tab[i]*q/qmax;

            }

            else if(newq < qmin){

                bits_sum -= bits_tab[i];

                cplx_sum -= cplx_tab[i]*q/qmin;

            }

        }

    }

   

    for(i=0; i<s->mb_num; i++){

        float newq= q*cplx_tab[i]/bits_tab[i];

        int intq;



        if(s->flags&CODEC_FLAG_NORMALIZE_AQP){

            newq*= bits_sum/cplx_sum;

        }



        if(i && ABS(pic->qscale_table[i-1] - newq)<0.75)

            intq= pic->qscale_table[i-1];

        else

            intq= (int)(newq + 0.5);



        if     (intq > qmax) intq= qmax;

        else if(intq < qmin) intq= qmin;

//if(i%s->mb_width==0) printf("\n");

//printf("%2d%3d ", intq, ff_sqrt(s->mc_mb_var[i]));

        pic->qscale_table[i]= intq;

    }

}
