void ff_h263_encode_mb(MpegEncContext * s,

                       int16_t block[6][64],

                       int motion_x, int motion_y)

{

    int cbpc, cbpy, i, cbp, pred_x, pred_y;

    int16_t pred_dc;

    int16_t rec_intradc[6];

    int16_t *dc_ptr[6];

    const int interleaved_stats= (s->flags&CODEC_FLAG_PASS1);



    if (!s->mb_intra) {

        /* compute cbp */

        cbp= get_p_cbp(s, block, motion_x, motion_y);



        if ((cbp | motion_x | motion_y | s->dquant | (s->mv_type - MV_TYPE_16X16)) == 0) {

            /* skip macroblock */

            put_bits(&s->pb, 1, 1);

            if(interleaved_stats){

                s->misc_bits++;

                s->last_bits++;

            }

            s->skip_count++;



            return;

        }

        put_bits(&s->pb, 1, 0);         /* mb coded */



        cbpc = cbp & 3;

        cbpy = cbp >> 2;

        if(s->alt_inter_vlc==0 || cbpc!=3)

            cbpy ^= 0xF;

        if(s->dquant) cbpc+= 8;

        if(s->mv_type==MV_TYPE_16X16){

            put_bits(&s->pb,

                    ff_h263_inter_MCBPC_bits[cbpc],

                    ff_h263_inter_MCBPC_code[cbpc]);



            put_bits(&s->pb, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);

            if(s->dquant)

                put_bits(&s->pb, 2, dquant_code[s->dquant+2]);



            if(interleaved_stats){

                s->misc_bits+= get_bits_diff(s);

            }



            /* motion vectors: 16x16 mode */

            ff_h263_pred_motion(s, 0, 0, &pred_x, &pred_y);



            if (!s->umvplus) {

                ff_h263_encode_motion_vector(s, motion_x - pred_x,

                                                motion_y - pred_y, 1);

            }

            else {

                h263p_encode_umotion(s, motion_x - pred_x);

                h263p_encode_umotion(s, motion_y - pred_y);

                if (((motion_x - pred_x) == 1) && ((motion_y - pred_y) == 1))

                    /* To prevent Start Code emulation */

                    put_bits(&s->pb,1,1);

            }

        }else{

            put_bits(&s->pb,

                    ff_h263_inter_MCBPC_bits[cbpc+16],

                    ff_h263_inter_MCBPC_code[cbpc+16]);

            put_bits(&s->pb, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);

            if(s->dquant)

                put_bits(&s->pb, 2, dquant_code[s->dquant+2]);



            if(interleaved_stats){

                s->misc_bits+= get_bits_diff(s);

            }



            for(i=0; i<4; i++){

                /* motion vectors: 8x8 mode*/

                ff_h263_pred_motion(s, i, 0, &pred_x, &pred_y);



                motion_x = s->current_picture.motion_val[0][s->block_index[i]][0];

                motion_y = s->current_picture.motion_val[0][s->block_index[i]][1];

                if (!s->umvplus) {

                    ff_h263_encode_motion_vector(s, motion_x - pred_x,

                                                    motion_y - pred_y, 1);

                }

                else {

                    h263p_encode_umotion(s, motion_x - pred_x);

                    h263p_encode_umotion(s, motion_y - pred_y);

                    if (((motion_x - pred_x) == 1) && ((motion_y - pred_y) == 1))

                        /* To prevent Start Code emulation */

                        put_bits(&s->pb,1,1);

                }

            }

        }



        if(interleaved_stats){

            s->mv_bits+= get_bits_diff(s);

        }

    } else {

        assert(s->mb_intra);



        cbp = 0;

        if (s->h263_aic) {

            /* Predict DC */

            for(i=0; i<6; i++) {

                int16_t level = block[i][0];

                int scale;



                if(i<4) scale= s->y_dc_scale;

                else    scale= s->c_dc_scale;



                pred_dc = ff_h263_pred_dc(s, i, &dc_ptr[i]);

                level -= pred_dc;

                /* Quant */

                if (level >= 0)

                    level = (level + (scale>>1))/scale;

                else

                    level = (level - (scale>>1))/scale;



                /* AIC can change CBP */

                if (level == 0 && s->block_last_index[i] == 0)

                    s->block_last_index[i] = -1;



                if(!s->modified_quant){

                    if (level < -127)

                        level = -127;

                    else if (level > 127)

                        level = 127;

                }



                block[i][0] = level;

                /* Reconstruction */

                rec_intradc[i] = scale*level + pred_dc;

                /* Oddify */

                rec_intradc[i] |= 1;

                //if ((rec_intradc[i] % 2) == 0)

                //    rec_intradc[i]++;

                /* Clipping */

                if (rec_intradc[i] < 0)

                    rec_intradc[i] = 0;

                else if (rec_intradc[i] > 2047)

                    rec_intradc[i] = 2047;



                /* Update AC/DC tables */

                *dc_ptr[i] = rec_intradc[i];

                if (s->block_last_index[i] >= 0)

                    cbp |= 1 << (5 - i);

            }

        }else{

            for(i=0; i<6; i++) {

                /* compute cbp */

                if (s->block_last_index[i] >= 1)

                    cbp |= 1 << (5 - i);

            }

        }



        cbpc = cbp & 3;

        if (s->pict_type == AV_PICTURE_TYPE_I) {

            if(s->dquant) cbpc+=4;

            put_bits(&s->pb,

                ff_h263_intra_MCBPC_bits[cbpc],

                ff_h263_intra_MCBPC_code[cbpc]);

        } else {

            if(s->dquant) cbpc+=8;

            put_bits(&s->pb, 1, 0);     /* mb coded */

            put_bits(&s->pb,

                ff_h263_inter_MCBPC_bits[cbpc + 4],

                ff_h263_inter_MCBPC_code[cbpc + 4]);

        }

        if (s->h263_aic) {

            /* XXX: currently, we do not try to use ac prediction */

            put_bits(&s->pb, 1, 0);     /* no AC prediction */

        }

        cbpy = cbp >> 2;

        put_bits(&s->pb, ff_h263_cbpy_tab[cbpy][1], ff_h263_cbpy_tab[cbpy][0]);

        if(s->dquant)

            put_bits(&s->pb, 2, dquant_code[s->dquant+2]);



        if(interleaved_stats){

            s->misc_bits+= get_bits_diff(s);

        }

    }



    for(i=0; i<6; i++) {

        /* encode each block */

        h263_encode_block(s, block[i], i);



        /* Update INTRADC for decoding */

        if (s->h263_aic && s->mb_intra) {

            block[i][0] = rec_intradc[i];



        }

    }



    if(interleaved_stats){

        if (!s->mb_intra) {

            s->p_tex_bits+= get_bits_diff(s);

            s->f_count++;

        }else{

            s->i_tex_bits+= get_bits_diff(s);

            s->i_count++;

        }

    }

}
