static int decode_residual(H264Context *h, GetBitContext *gb, DCTELEM *block, int n, const uint8_t *scantable, int qp, int max_coeff){

    MpegEncContext * const s = &h->s;

    const uint16_t *qmul= dequant_coeff[qp];

    static const int coeff_token_table_index[17]= {0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};

    int level[16], run[16];

    int suffix_length, zeros_left, coeff_num, coeff_token, total_coeff, i, trailing_ones;



    //FIXME put trailing_onex into the context



    if(n == CHROMA_DC_BLOCK_INDEX){

        coeff_token= get_vlc2(gb, chroma_dc_coeff_token_vlc.table, CHROMA_DC_COEFF_TOKEN_VLC_BITS, 1);

        total_coeff= coeff_token>>2;

    }else{    

        if(n == LUMA_DC_BLOCK_INDEX){

            total_coeff= pred_non_zero_count(h, 0);

            coeff_token= get_vlc2(gb, coeff_token_vlc[ coeff_token_table_index[total_coeff] ].table, COEFF_TOKEN_VLC_BITS, 2);

            total_coeff= coeff_token>>2;

        }else{

            total_coeff= pred_non_zero_count(h, n);

            coeff_token= get_vlc2(gb, coeff_token_vlc[ coeff_token_table_index[total_coeff] ].table, COEFF_TOKEN_VLC_BITS, 2);

            total_coeff= coeff_token>>2;

            h->non_zero_count_cache[ scan8[n] ]= total_coeff;

        }

    }



    //FIXME set last_non_zero?



    if(total_coeff==0)

        return 0;

        

    trailing_ones= coeff_token&3;

    tprintf("trailing:%d, total:%d\n", trailing_ones, total_coeff);

    assert(total_coeff<=16);

    

    for(i=0; i<trailing_ones; i++){

        level[i]= 1 - 2*get_bits1(gb);

    }



    suffix_length= total_coeff > 10 && trailing_ones < 3;



    for(; i<total_coeff; i++){

        const int prefix= get_level_prefix(gb);

        int level_code, mask;



        if(prefix<14){ //FIXME try to build a large unified VLC table for all this

            if(suffix_length)

                level_code= (prefix<<suffix_length) + get_bits(gb, suffix_length); //part

            else

                level_code= (prefix<<suffix_length); //part

        }else if(prefix==14){

            if(suffix_length)

                level_code= (prefix<<suffix_length) + get_bits(gb, suffix_length); //part

            else

                level_code= prefix + get_bits(gb, 4); //part

        }else if(prefix==15){

            level_code= (prefix<<suffix_length) + get_bits(gb, 12); //part

            if(suffix_length==0) level_code+=15; //FIXME doesnt make (much)sense

        }else{

            av_log(h->s.avctx, AV_LOG_ERROR, "prefix too large at %d %d\n", s->mb_x, s->mb_y);

            return -1;

        }



        if(i==trailing_ones && i<3) level_code+= 2; //FIXME split first iteration



        mask= -(level_code&1);

        level[i]= (((2+level_code)>>1) ^ mask) - mask;



        if(suffix_length==0) suffix_length=1; //FIXME split first iteration



#if 1

        if(ABS(level[i]) > (3<<(suffix_length-1)) && suffix_length<6) suffix_length++;

#else        

        if((2+level_code)>>1) > (3<<(suffix_length-1)) && suffix_length<6) suffix_length++;

        ? == prefix > 2 or sth

#endif

        tprintf("level: %d suffix_length:%d\n", level[i], suffix_length);

    }



    if(total_coeff == max_coeff)

        zeros_left=0;

    else{

        if(n == CHROMA_DC_BLOCK_INDEX)

            zeros_left= get_vlc2(gb, chroma_dc_total_zeros_vlc[ total_coeff-1 ].table, CHROMA_DC_TOTAL_ZEROS_VLC_BITS, 1);

        else

            zeros_left= get_vlc2(gb, total_zeros_vlc[ total_coeff-1 ].table, TOTAL_ZEROS_VLC_BITS, 1);

    }

    

    for(i=0; i<total_coeff-1; i++){

        if(zeros_left <=0)

            break;

        else if(zeros_left < 7){

            run[i]= get_vlc2(gb, run_vlc[zeros_left-1].table, RUN_VLC_BITS, 1);

        }else{

            run[i]= get_vlc2(gb, run7_vlc.table, RUN7_VLC_BITS, 2);

        }

        zeros_left -= run[i];

    }



    if(zeros_left<0){

        av_log(h->s.avctx, AV_LOG_ERROR, "negative number of zero coeffs at %d %d\n", s->mb_x, s->mb_y);

        return -1;

    }

    

    for(; i<total_coeff-1; i++){

        run[i]= 0;

    }



    run[i]= zeros_left;



    coeff_num=-1;

    if(n > 24){

        for(i=total_coeff-1; i>=0; i--){ //FIXME merge into rundecode?

            int j;



            coeff_num += run[i] + 1; //FIXME add 1 earlier ?

            j= scantable[ coeff_num ];



            block[j]= level[i];

        }

    }else{

        for(i=total_coeff-1; i>=0; i--){ //FIXME merge into  rundecode?

            int j;



            coeff_num += run[i] + 1; //FIXME add 1 earlier ?

            j= scantable[ coeff_num ];



            block[j]= level[i] * qmul[j];

//            printf("%d %d  ", block[j], qmul[j]);

        }

    }

    return 0;

}
