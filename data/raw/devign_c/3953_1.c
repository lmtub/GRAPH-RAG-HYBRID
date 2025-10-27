static int x8_decode_intra_mb(IntraX8Context* const w, const int chroma){

    MpegEncContext * const s= w->s;



    uint8_t * scantable;

    int final,run,level;

    int ac_mode,dc_mode,est_run,dc_level;

    int pos,n;

    int zeros_only;

    int use_quant_matrix;

    int sign;



    assert(w->orient<12);

    s->dsp.clear_block(s->block[0]);



    if(chroma){

        dc_mode=2;

    }else{

        dc_mode=!!w->est_run;//0,1

    }



    if(x8_get_dc_rlf(w, dc_mode, &dc_level, &final)) return -1;

    n=0;

    zeros_only=0;

    if(!final){//decode ac

        use_quant_matrix=w->use_quant_matrix;

        if(chroma){

            ac_mode = 1;

            est_run = 64;//not used

        }else{

            if (w->raw_orient < 3){

                use_quant_matrix = 0;

            }

            if(w->raw_orient > 4){

                ac_mode = 0;

                est_run = 64;

            }else{

                if(w->est_run > 1){

                    ac_mode = 2;

                    est_run=w->est_run;

                }else{

                    ac_mode = 3;

                    est_run = 64;

                }

            }

        }

        x8_select_ac_table(w,ac_mode);

        /*scantable_selector[12]={0,2,0,1,1,1,0,2,2,0,1,2};<-

        -> 10'01' 00'10' 10'00' 01'01' 01'00' 10'00 =>0x928548 */

        scantable = w->scantable[ (0x928548>>(2*w->orient))&3 ].permutated;

        pos=0;

        do {

            n++;

            if( n >= est_run ){

                ac_mode=3;

                x8_select_ac_table(w,3);

            }



            x8_get_ac_rlf(w,ac_mode,&run,&level,&final);



            pos+=run+1;

            if(pos>63){

                //this also handles vlc error in x8_get_ac_rlf

                return -1;

            }

            level= (level+1) * w->dquant;

            level+= w->qsum;



            sign = - get_bits1(&s->gb);

            level = (level ^ sign) - sign;



            if(use_quant_matrix){

                level = (level*quant_table[pos])>>8;

            }

            s->block[0][ scantable[pos] ]=level;

        }while(!final);



        s->block_last_index[0]=pos;

    }else{//DC only

        s->block_last_index[0]=0;

        if(w->flat_dc && ((unsigned)(dc_level+1)) < 3){//[-1;1]

            int32_t divide_quant= !chroma ? w->divide_quant_dc_luma:

                                            w->divide_quant_dc_chroma;

            int32_t dc_quant    = !chroma ? w->quant:

                                            w->quant_dc_chroma;



            //original intent dc_level+=predicted_dc/quant; but it got lost somewhere in the rounding

            dc_level+= (w->predicted_dc*divide_quant + (1<<12) )>>13;



            dsp_x8_put_solidcolor( av_clip_uint8((dc_level*dc_quant+4)>>3),

                                   s->dest[chroma], s->current_picture.f.linesize[!!chroma]);



            goto block_placed;

        }

        zeros_only = (dc_level == 0);

    }

    if(!chroma){

        s->block[0][0] = dc_level*w->quant;

    }else{

        s->block[0][0] = dc_level*w->quant_dc_chroma;

    }



    //there is !zero_only check in the original, but dc_level check is enough

    if( (unsigned int)(dc_level+1) >= 3 && (w->edges&3) != 3 ){

        int direction;

        /*ac_comp_direction[orient] = { 0, 3, 3, 1, 1, 0, 0, 0, 2, 2, 2, 1 };<-

        -> 01'10' 10'10' 00'00' 00'01' 01'11' 11'00 =>0x6A017C */

        direction= (0x6A017C>>(w->orient*2))&3;

        if (direction != 3){

            x8_ac_compensation(w, direction, s->block[0][0]);//modify block_last[]

        }

    }



    if(w->flat_dc){

        dsp_x8_put_solidcolor(w->predicted_dc, s->dest[chroma], s->current_picture.f.linesize[!!chroma]);

    }else{

        w->dsp.spatial_compensation[w->orient]( s->edge_emu_buffer,

                                            s->dest[chroma],

                                            s->current_picture.f.linesize[!!chroma] );

    }

    if(!zeros_only)

        s->dsp.idct_add ( s->dest[chroma],

                          s->current_picture.f.linesize[!!chroma],

                          s->block[0] );



block_placed:



    if(!chroma){

        x8_update_predictions(w,w->orient,n);

    }



    if(s->loop_filter){

        uint8_t* ptr = s->dest[chroma];

        int linesize = s->current_picture.f.linesize[!!chroma];



        if(!( (w->edges&2) || ( zeros_only && (w->orient|4)==4 ) )){

            w->dsp.h_loop_filter(ptr, linesize, w->quant);

        }

        if(!( (w->edges&1) || ( zeros_only && (w->orient|8)==8 ) )){

            w->dsp.v_loop_filter(ptr, linesize, w->quant);

        }

    }

    return 0;

}
