static int decode_slice(H264Context *h){

    MpegEncContext * const s = &h->s;

    const int part_mask= s->partitioned_frame ? (AC_END|AC_ERROR) : 0x7F;



    s->mb_skip_run= -1;



    if( h->pps.cabac ) {

        int i;



        /* realign */

        align_get_bits( &s->gb );



        /* init cabac */

        ff_init_cabac_states( &h->cabac, ff_h264_lps_range, ff_h264_mps_state, ff_h264_lps_state, 64 );

        ff_init_cabac_decoder( &h->cabac,

                               s->gb.buffer + get_bits_count(&s->gb)/8,

                               ( s->gb.size_in_bits - get_bits_count(&s->gb) + 7)/8);

        /* calculate pre-state */

        for( i= 0; i < 460; i++ ) {

            int pre;

            if( h->slice_type == I_TYPE )

                pre = clip( ((cabac_context_init_I[i][0] * s->qscale) >>4 ) + cabac_context_init_I[i][1], 1, 126 );

            else

                pre = clip( ((cabac_context_init_PB[h->cabac_init_idc][i][0] * s->qscale) >>4 ) + cabac_context_init_PB[h->cabac_init_idc][i][1], 1, 126 );



            if( pre <= 63 )

                h->cabac_state[i] = 2 * ( 63 - pre ) + 0;

            else

                h->cabac_state[i] = 2 * ( pre - 64 ) + 1;

        }



        for(;;){

            int ret = decode_mb_cabac(h);

            int eos;



            if(ret>=0) hl_decode_mb(h);



            /* XXX: useless as decode_mb_cabac it doesn't support that ... */

            if( ret >= 0 && h->mb_aff_frame ) { //FIXME optimal? or let mb_decode decode 16x32 ?

                s->mb_y++;



                if(ret>=0) ret = decode_mb_cabac(h);



                hl_decode_mb(h);

                s->mb_y--;

            }

            eos = get_cabac_terminate( &h->cabac );



            if( ret < 0 || h->cabac.bytestream > h->cabac.bytestream_end + 1) {

                av_log(h->s.avctx, AV_LOG_ERROR, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);

                return -1;

            }



            if( ++s->mb_x >= s->mb_width ) {

                s->mb_x = 0;

                ff_draw_horiz_band(s, 16*s->mb_y, 16);

                ++s->mb_y;

                if(h->mb_aff_frame) {

                    ++s->mb_y;

                }

            }



            if( eos || s->mb_y >= s->mb_height ) {

                tprintf("slice end %d %d\n", get_bits_count(&s->gb), s->gb.size_in_bits);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);

                return 0;

            }

        }



    } else {

        for(;;){

            int ret = decode_mb_cavlc(h);



            if(ret>=0) hl_decode_mb(h);



            if(ret>=0 && h->mb_aff_frame){ //FIXME optimal? or let mb_decode decode 16x32 ?

                s->mb_y++;

                ret = decode_mb_cavlc(h);



                if(ret>=0) hl_decode_mb(h);

                s->mb_y--;

            }



            if(ret<0){

                av_log(h->s.avctx, AV_LOG_ERROR, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                return -1;

            }



            if(++s->mb_x >= s->mb_width){

                s->mb_x=0;

                ff_draw_horiz_band(s, 16*s->mb_y, 16);

                ++s->mb_y;

                if(h->mb_aff_frame) {

                    ++s->mb_y;

                }

                if(s->mb_y >= s->mb_height){

                    tprintf("slice end %d %d\n", get_bits_count(&s->gb), s->gb.size_in_bits);



                    if(get_bits_count(&s->gb) == s->gb.size_in_bits ) {

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return 0;

                    }else{

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return -1;

                    }

                }

            }



            if(get_bits_count(&s->gb) >= s->gb.size_in_bits && s->mb_skip_run<=0){

                tprintf("slice end %d %d\n", get_bits_count(&s->gb), s->gb.size_in_bits);

                if(get_bits_count(&s->gb) == s->gb.size_in_bits ){

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                    return 0;

                }else{

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                    return -1;

                }

            }

        }

    }



#if 0

    for(;s->mb_y < s->mb_height; s->mb_y++){

        for(;s->mb_x < s->mb_width; s->mb_x++){

            int ret= decode_mb(h);

            

            hl_decode_mb(h);



            if(ret<0){

                fprintf(stderr, "error while decoding MB %d %d\n", s->mb_x, s->mb_y);

                ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                return -1;

            }

        

            if(++s->mb_x >= s->mb_width){

                s->mb_x=0;

                if(++s->mb_y >= s->mb_height){

                    if(get_bits_count(s->gb) == s->gb.size_in_bits){

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return 0;

                    }else{

                        ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                        return -1;

                    }

                }

            }

        

            if(get_bits_count(s->?gb) >= s->gb?.size_in_bits){

                if(get_bits_count(s->gb) == s->gb.size_in_bits){

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x-1, s->mb_y, (AC_END|DC_END|MV_END)&part_mask);



                    return 0;

                }else{

                    ff_er_add_slice(s, s->resync_mb_x, s->resync_mb_y, s->mb_x, s->mb_y, (AC_ERROR|DC_ERROR|MV_ERROR)&part_mask);



                    return -1;

                }

            }

        }

        s->mb_x=0;

        ff_draw_horiz_band(s, 16*s->mb_y, 16);

    }

#endif

    return -1; //not reached

}
