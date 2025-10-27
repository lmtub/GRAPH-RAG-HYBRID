static int decode_ref_pic_marking(H264Context *h){

    MpegEncContext * const s = &h->s;

    int i;



    if(h->nal_unit_type == NAL_IDR_SLICE){ //FIXME fields

        s->broken_link= get_bits1(&s->gb) -1;

        h->mmco[0].long_index= get_bits1(&s->gb) - 1; // current_long_term_idx

        if(h->mmco[0].long_index == -1)

            h->mmco_index= 0;

        else{

            h->mmco[0].opcode= MMCO_LONG;

            h->mmco_index= 1;

        }

    }else{

        if(get_bits1(&s->gb)){ // adaptive_ref_pic_marking_mode_flag

            for(i= 0; i<MAX_MMCO_COUNT; i++) {

                MMCOOpcode opcode= get_ue_golomb(&s->gb);;



                h->mmco[i].opcode= opcode;

                if(opcode==MMCO_SHORT2UNUSED || opcode==MMCO_SHORT2LONG){

                    h->mmco[i].short_frame_num= (h->frame_num - get_ue_golomb(&s->gb) - 1) & ((1<<h->sps.log2_max_frame_num)-1); //FIXME fields

/*                    if(h->mmco[i].short_frame_num >= h->short_ref_count || h->short_ref[ h->mmco[i].short_frame_num ] == NULL){

                        av_log(s->avctx, AV_LOG_ERROR, "illegal short ref in memory management control operation %d\n", mmco);

                        return -1;

                    }*/

                }

                if(opcode==MMCO_SHORT2LONG || opcode==MMCO_LONG2UNUSED || opcode==MMCO_LONG || opcode==MMCO_SET_MAX_LONG){

                    h->mmco[i].long_index= get_ue_golomb(&s->gb);

                    if(/*h->mmco[i].long_index >= h->long_ref_count || h->long_ref[ h->mmco[i].long_index ] == NULL*/ h->mmco[i].long_index >= 16){

                        av_log(h->s.avctx, AV_LOG_ERROR, "illegal long ref in memory management control operation %d\n", opcode);

                        return -1;

                    }

                }



                if(opcode > MMCO_LONG){

                    av_log(h->s.avctx, AV_LOG_ERROR, "illegal memory management control operation %d\n", opcode);

                    return -1;

                }

                if(opcode == MMCO_END)

                    break;

            }

            h->mmco_index= i;

        }else{

            assert(h->long_ref_count + h->short_ref_count <= h->sps.ref_frame_count);



            if(h->long_ref_count + h->short_ref_count == h->sps.ref_frame_count){ //FIXME fields

                h->mmco[0].opcode= MMCO_SHORT2UNUSED;

                h->mmco[0].short_frame_num= h->short_ref[ h->short_ref_count - 1 ]->frame_num;

                h->mmco_index= 1;

            }else

                h->mmco_index= 0;

        }

    }



    return 0;

}
