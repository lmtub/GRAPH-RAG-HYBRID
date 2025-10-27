static int decode_nal_units(H264Context *h, uint8_t *buf, int buf_size){

    MpegEncContext * const s = &h->s;

    AVCodecContext * const avctx= s->avctx;

    int buf_index=0;

#if 0

    int i;

    for(i=0; i<32; i++){

        printf("%X ", buf[i]);

    }

#endif

    for(;;){

        int consumed;

        int dst_length;

        int bit_length;

        uint8_t *ptr;

        

        // start code prefix search

        for(; buf_index + 3 < buf_size; buf_index++){

            // this should allways succeed in the first iteration

            if(buf[buf_index] == 0 && buf[buf_index+1] == 0 && buf[buf_index+2] == 1)

                break;

        }

        

        if(buf_index+3 >= buf_size) break;

        

        buf_index+=3;

        

        ptr= decode_nal(h, buf + buf_index, &dst_length, &consumed, buf_size - buf_index);

        if(ptr[dst_length - 1] == 0) dst_length--;

        bit_length= 8*dst_length - decode_rbsp_trailing(ptr + dst_length - 1);



        if(s->avctx->debug&FF_DEBUG_STARTCODE){

            av_log(h->s.avctx, AV_LOG_DEBUG, "NAL %d at %d length %d\n", h->nal_unit_type, buf_index, dst_length);

        }

        

        buf_index += consumed;



        if(h->nal_ref_idc < s->hurry_up)

            continue;

        

        switch(h->nal_unit_type){

        case NAL_IDR_SLICE:

            idr(h); //FIXME ensure we dont loose some frames if there is reordering

        case NAL_SLICE:

            init_get_bits(&s->gb, ptr, bit_length);

            h->intra_gb_ptr=

            h->inter_gb_ptr= &s->gb;

            s->data_partitioning = 0;

            

            if(decode_slice_header(h) < 0) return -1;

            if(h->redundant_pic_count==0)

                decode_slice(h);

            break;

        case NAL_DPA:

            init_get_bits(&s->gb, ptr, bit_length);

            h->intra_gb_ptr=

            h->inter_gb_ptr= NULL;

            s->data_partitioning = 1;

            

            if(decode_slice_header(h) < 0) return -1;

            break;

        case NAL_DPB:

            init_get_bits(&h->intra_gb, ptr, bit_length);

            h->intra_gb_ptr= &h->intra_gb;

            break;

        case NAL_DPC:

            init_get_bits(&h->inter_gb, ptr, bit_length);

            h->inter_gb_ptr= &h->inter_gb;



            if(h->redundant_pic_count==0 && h->intra_gb_ptr && s->data_partitioning)

                decode_slice(h);

            break;

        case NAL_SEI:

            break;

        case NAL_SPS:

            init_get_bits(&s->gb, ptr, bit_length);

            decode_seq_parameter_set(h);

            

            if(s->flags& CODEC_FLAG_LOW_DELAY)

                s->low_delay=1;

      

            avctx->has_b_frames= !s->low_delay;

            break;

        case NAL_PPS:

            init_get_bits(&s->gb, ptr, bit_length);

            

            decode_picture_parameter_set(h);



            break;

        case NAL_PICTURE_DELIMITER:

            break;

        case NAL_FILTER_DATA:

            break;

        }        



        //FIXME move after where irt is set

        s->current_picture.pict_type= s->pict_type;

        s->current_picture.key_frame= s->pict_type == I_TYPE;

    }

    

    if(!s->current_picture_ptr) return buf_index; //no frame

    

    h->prev_frame_num_offset= h->frame_num_offset;

    h->prev_frame_num= h->frame_num;

    if(s->current_picture_ptr->reference){

        h->prev_poc_msb= h->poc_msb;

        h->prev_poc_lsb= h->poc_lsb;

    }

    if(s->current_picture_ptr->reference)

        execute_ref_pic_marking(h, h->mmco, h->mmco_index);

    else

        assert(h->mmco_index==0);



    ff_er_frame_end(s);



    if( h->disable_deblocking_filter_idc != 1 ) {

        filter_frame( h );

    }



    MPV_frame_end(s);



    return buf_index;

}
