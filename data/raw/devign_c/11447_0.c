int ff_h263_decode_mb(MpegEncContext *s,

                      DCTELEM block[6][64])

{

    int cbpc, cbpy, i, cbp, pred_x, pred_y, mx, my, dquant;

    INT16 *mot_val;

    static INT8 quant_tab[4] = { -1, -2, 1, 2 };



    s->error_status_table[s->mb_x + s->mb_y*s->mb_width]= 0;



    if(s->mb_x==0) PRINT_MB_TYPE("\n");



    if (s->pict_type == P_TYPE || s->pict_type==S_TYPE) {

        if (get_bits1(&s->gb)) {

            /* skip mb */

            s->mb_intra = 0;

            for(i=0;i<6;i++)

                s->block_last_index[i] = -1;

            s->mv_dir = MV_DIR_FORWARD;

            s->mv_type = MV_TYPE_16X16;

            if(s->pict_type==S_TYPE && s->vol_sprite_usage==GMC_SPRITE){

                PRINT_MB_TYPE("G");

                s->mcsel=1;

                s->mv[0][0][0]= get_amv(s, 0);

                s->mv[0][0][1]= get_amv(s, 1);



                s->mb_skiped = 0;

            }else{

                PRINT_MB_TYPE("S");

                s->mcsel=0;

                s->mv[0][0][0] = 0;

                s->mv[0][0][1] = 0;

                s->mb_skiped = 1;

            }

            goto end;

        }

        cbpc = get_vlc2(&s->gb, inter_MCBPC_vlc.table, INTER_MCBPC_VLC_BITS, 2);

        //fprintf(stderr, "\tCBPC: %d", cbpc);

        if (cbpc < 0)

            return -1;

        if (cbpc > 20)

            cbpc+=3;

        else if (cbpc == 20)

            fprintf(stderr, "Stuffing !");

        

        dquant = cbpc & 8;

        s->mb_intra = ((cbpc & 4) != 0);

        if (s->mb_intra) goto intra;

        

        if(s->pict_type==S_TYPE && s->vol_sprite_usage==GMC_SPRITE && (cbpc & 16) == 0)

            s->mcsel= get_bits1(&s->gb);

        else s->mcsel= 0;

        cbpy = get_vlc2(&s->gb, cbpy_vlc.table, CBPY_VLC_BITS, 1);

        cbp = (cbpc & 3) | ((cbpy ^ 0xf) << 2);

        if (dquant) {

            change_qscale(s, quant_tab[get_bits(&s->gb, 2)]);

        }

        if((!s->progressive_sequence) && (cbp || (s->workaround_bugs&FF_BUG_XVID_ILACE)))

            s->interlaced_dct= get_bits1(&s->gb);

        

        s->mv_dir = MV_DIR_FORWARD;

        if ((cbpc & 16) == 0) {

            if(s->mcsel){

                PRINT_MB_TYPE("G");

                /* 16x16 global motion prediction */

                s->mv_type = MV_TYPE_16X16;

                mx= get_amv(s, 0);

                my= get_amv(s, 1);

                s->mv[0][0][0] = mx;

                s->mv[0][0][1] = my;

            }else if((!s->progressive_sequence) && get_bits1(&s->gb)){

                PRINT_MB_TYPE("f");

                /* 16x8 field motion prediction */

                s->mv_type= MV_TYPE_FIELD;



                s->field_select[0][0]= get_bits1(&s->gb);

                s->field_select[0][1]= get_bits1(&s->gb);



                h263_pred_motion(s, 0, &pred_x, &pred_y);

                

                for(i=0; i<2; i++){

                    mx = h263_decode_motion(s, pred_x, s->f_code);

                    if (mx >= 0xffff)

                        return -1;

            

                    my = h263_decode_motion(s, pred_y/2, s->f_code);

                    if (my >= 0xffff)

                        return -1;



                    s->mv[0][i][0] = mx;

                    s->mv[0][i][1] = my;

                }

            }else{

                PRINT_MB_TYPE("P");

                /* 16x16 motion prediction */

                s->mv_type = MV_TYPE_16X16;

                h263_pred_motion(s, 0, &pred_x, &pred_y);

                if (s->umvplus_dec)

                   mx = h263p_decode_umotion(s, pred_x);

                else

                   mx = h263_decode_motion(s, pred_x, s->f_code);

            

                if (mx >= 0xffff)

                    return -1;

            

                if (s->umvplus_dec)

                   my = h263p_decode_umotion(s, pred_y);

                else

                   my = h263_decode_motion(s, pred_y, s->f_code);

            

                if (my >= 0xffff)

                    return -1;

                s->mv[0][0][0] = mx;

                s->mv[0][0][1] = my;



                if (s->umvplus_dec && (mx - pred_x) == 1 && (my - pred_y) == 1)

                   skip_bits1(&s->gb); /* Bit stuffing to prevent PSC */                   

            }

        } else {

            PRINT_MB_TYPE("4");

            s->mv_type = MV_TYPE_8X8;

            for(i=0;i<4;i++) {

                mot_val = h263_pred_motion(s, i, &pred_x, &pred_y);

                if (s->umvplus_dec)

                  mx = h263p_decode_umotion(s, pred_x);

                else

                  mx = h263_decode_motion(s, pred_x, s->f_code);

                if (mx >= 0xffff)

                    return -1;

                

                if (s->umvplus_dec)

                  my = h263p_decode_umotion(s, pred_y);

                else    

                  my = h263_decode_motion(s, pred_y, s->f_code);

                if (my >= 0xffff)

                    return -1;

                s->mv[0][i][0] = mx;

                s->mv[0][i][1] = my;

                if (s->umvplus_dec && (mx - pred_x) == 1 && (my - pred_y) == 1)

                  skip_bits1(&s->gb); /* Bit stuffing to prevent PSC */

                mot_val[0] = mx;

                mot_val[1] = my;

            }

        }

    } else if(s->pict_type==B_TYPE) {

        int modb1; // first bit of modb

        int modb2; // second bit of modb

        int mb_type;

        int xy;



        s->mb_intra = 0; //B-frames never contain intra blocks

        s->mcsel=0;      //     ...               true gmc blocks



        if(s->mb_x==0){

            for(i=0; i<2; i++){

                s->last_mv[i][0][0]= 

                s->last_mv[i][0][1]= 

                s->last_mv[i][1][0]= 

                s->last_mv[i][1][1]= 0;

            }

        }



        /* if we skipped it in the future P Frame than skip it now too */

        s->mb_skiped= s->next_picture.mbskip_table[s->mb_y * s->mb_width + s->mb_x]; // Note, skiptab=0 if last was GMC



        if(s->mb_skiped){

                /* skip mb */

            for(i=0;i<6;i++)

                s->block_last_index[i] = -1;



            s->mv_dir = MV_DIR_FORWARD;

            s->mv_type = MV_TYPE_16X16;

            s->mv[0][0][0] = 0;

            s->mv[0][0][1] = 0;

            s->mv[1][0][0] = 0;

            s->mv[1][0][1] = 0;

            PRINT_MB_TYPE("s");

            goto end;

        }



        modb1= get_bits1(&s->gb); 

        if(modb1){

            mb_type=4; //like MB_TYPE_B_DIRECT but no vectors coded

            cbp=0;

        }else{

            int field_mv;

        

            modb2= get_bits1(&s->gb);

            mb_type= get_vlc2(&s->gb, mb_type_b_vlc.table, MB_TYPE_B_VLC_BITS, 1);

            if(modb2) cbp= 0;

            else      cbp= get_bits(&s->gb, 6);



            if (mb_type!=MB_TYPE_B_DIRECT && cbp) {

                if(get_bits1(&s->gb)){

                    change_qscale(s, get_bits1(&s->gb)*4 - 2);

                }

            }

            field_mv=0;



            if(!s->progressive_sequence){

                if(cbp)

                    s->interlaced_dct= get_bits1(&s->gb);



                if(mb_type!=MB_TYPE_B_DIRECT && get_bits1(&s->gb)){

                    field_mv=1;



                    if(mb_type!=MB_TYPE_B_BACKW){

                        s->field_select[0][0]= get_bits1(&s->gb);

                        s->field_select[0][1]= get_bits1(&s->gb);

                    }

                    if(mb_type!=MB_TYPE_B_FORW){

                        s->field_select[1][0]= get_bits1(&s->gb);

                        s->field_select[1][1]= get_bits1(&s->gb);

                    }

                }

            }



            s->mv_dir = 0;

            if(mb_type!=MB_TYPE_B_DIRECT && !field_mv){

                s->mv_type= MV_TYPE_16X16;

                if(mb_type!=MB_TYPE_B_BACKW){

                    s->mv_dir = MV_DIR_FORWARD;



                    mx = h263_decode_motion(s, s->last_mv[0][0][0], s->f_code);

                    my = h263_decode_motion(s, s->last_mv[0][0][1], s->f_code);

                    s->last_mv[0][1][0]= s->last_mv[0][0][0]= s->mv[0][0][0] = mx;

                    s->last_mv[0][1][1]= s->last_mv[0][0][1]= s->mv[0][0][1] = my;

                }

    

                if(mb_type!=MB_TYPE_B_FORW){

                    s->mv_dir |= MV_DIR_BACKWARD;



                    mx = h263_decode_motion(s, s->last_mv[1][0][0], s->b_code);

                    my = h263_decode_motion(s, s->last_mv[1][0][1], s->b_code);

                    s->last_mv[1][1][0]= s->last_mv[1][0][0]= s->mv[1][0][0] = mx;

                    s->last_mv[1][1][1]= s->last_mv[1][0][1]= s->mv[1][0][1] = my;

                }

                if(mb_type!=MB_TYPE_B_DIRECT)

                    PRINT_MB_TYPE(mb_type==MB_TYPE_B_FORW ? "F" : (mb_type==MB_TYPE_B_BACKW ? "B" : "T"));

            }else if(mb_type!=MB_TYPE_B_DIRECT){

                s->mv_type= MV_TYPE_FIELD;



                if(mb_type!=MB_TYPE_B_BACKW){

                    s->mv_dir = MV_DIR_FORWARD;

                

                    for(i=0; i<2; i++){

                        mx = h263_decode_motion(s, s->last_mv[0][i][0]  , s->f_code);

                        my = h263_decode_motion(s, s->last_mv[0][i][1]/2, s->f_code);

                        s->last_mv[0][i][0]=  s->mv[0][i][0] = mx;

                        s->last_mv[0][i][1]= (s->mv[0][i][1] = my)*2;

                    }

                }

    

                if(mb_type!=MB_TYPE_B_FORW){

                    s->mv_dir |= MV_DIR_BACKWARD;



                    for(i=0; i<2; i++){

                        mx = h263_decode_motion(s, s->last_mv[1][i][0]  , s->b_code);

                        my = h263_decode_motion(s, s->last_mv[1][i][1]/2, s->b_code);

                        s->last_mv[1][i][0]=  s->mv[1][i][0] = mx;

                        s->last_mv[1][i][1]= (s->mv[1][i][1] = my)*2;

                    }

                }

                if(mb_type!=MB_TYPE_B_DIRECT)

                    PRINT_MB_TYPE(mb_type==MB_TYPE_B_FORW ? "f" : (mb_type==MB_TYPE_B_BACKW ? "b" : "t"));

            }

        }

          

        if(mb_type==4 || mb_type==MB_TYPE_B_DIRECT){

            if(mb_type==4)

                mx=my=0;

            else{

                mx = h263_decode_motion(s, 0, 1);

                my = h263_decode_motion(s, 0, 1);

            }

 

            s->mv_dir = MV_DIR_FORWARD | MV_DIR_BACKWARD | MV_DIRECT;

            ff_mpeg4_set_direct_mv(s, mx, my);

        }

        

        if(mb_type<0 || mb_type>4){

            printf("illegal MB_type\n");

            return -1;

        }

    } else { /* I-Frame */

        cbpc = get_vlc2(&s->gb, intra_MCBPC_vlc.table, INTRA_MCBPC_VLC_BITS, 1);

        if (cbpc < 0)

            return -1;

        dquant = cbpc & 4;

        s->mb_intra = 1;

intra:

        s->ac_pred = 0;

        if (s->h263_pred || s->h263_aic) {

            s->ac_pred = get_bits1(&s->gb);

            if (s->ac_pred && s->h263_aic)

                s->h263_aic_dir = get_bits1(&s->gb);

        }

        PRINT_MB_TYPE(s->ac_pred ? "A" : "I");

        

        cbpy = get_vlc2(&s->gb, cbpy_vlc.table, CBPY_VLC_BITS, 1);

        if(cbpy<0) return -1;

        cbp = (cbpc & 3) | (cbpy << 2);

        if (dquant) {

            change_qscale(s, quant_tab[get_bits(&s->gb, 2)]);

        }

        

        if(!s->progressive_sequence)

            s->interlaced_dct= get_bits1(&s->gb);



        /* decode each block */

        if (s->h263_pred) {

            for (i = 0; i < 6; i++) {

                if (mpeg4_decode_block(s, block[i], i, cbp&32, 1) < 0)

                    return -1;

                cbp+=cbp;

            }

        } else {

            for (i = 0; i < 6; i++) {

                if (h263_decode_block(s, block[i], i, cbp&32) < 0)

                    return -1;

                cbp+=cbp;

            }

        }

        goto end;

    }



    /* decode each block */

    if (s->h263_pred) {

        for (i = 0; i < 6; i++) {

            if (mpeg4_decode_block(s, block[i], i, cbp&32, 0) < 0)

                return -1;

            cbp+=cbp;

        }

    } else {

        for (i = 0; i < 6; i++) {

            if (h263_decode_block(s, block[i], i, cbp&32) < 0)

                return -1;

            cbp+=cbp;

        }

    }

end:



        /* per-MB end of slice check */

    if(s->codec_id==CODEC_ID_MPEG4){

        if(mpeg4_is_resync(s)){

            if(s->pict_type==B_TYPE && s->next_picture.mbskip_table[s->mb_y * s->mb_width + s->mb_x+1])

                return SLICE_OK;

            return SLICE_END;

        }

    }else{

        int v= show_bits(&s->gb, 16);

    

        if(get_bits_count(&s->gb) + 16 > s->gb.size*8){

            v>>= get_bits_count(&s->gb) + 16 - s->gb.size*8;

        }



        if(v==0)

            return SLICE_END;

    }



    return SLICE_OK;     

}
