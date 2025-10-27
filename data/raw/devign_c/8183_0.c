static int decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    H264Context *h = avctx->priv_data;

    MpegEncContext *s = &h->s;

    AVFrame *pict = data;

    int buf_index;



    s->flags= avctx->flags;

    s->flags2= avctx->flags2;



   /* end of stream, output what is still in the buffers */

    if (buf_size == 0) {

        Picture *out;

        int i, out_idx;



//FIXME factorize this with the output code below

        out = h->delayed_pic[0];

        out_idx = 0;

        for(i=1; h->delayed_pic[i] && (h->delayed_pic[i]->poc && !h->delayed_pic[i]->key_frame); i++)

            if(h->delayed_pic[i]->poc < out->poc){

                out = h->delayed_pic[i];

                out_idx = i;

            }



        for(i=out_idx; h->delayed_pic[i]; i++)

            h->delayed_pic[i] = h->delayed_pic[i+1];



        if(out){

            *data_size = sizeof(AVFrame);

            *pict= *(AVFrame*)out;

        }



        return 0;

    }



    if(h->is_avc && !h->got_avcC) {

        int i, cnt, nalsize;

        unsigned char *p = avctx->extradata;

        if(avctx->extradata_size < 7) {

            av_log(avctx, AV_LOG_ERROR, "avcC too short\n");

            return -1;

        }

        if(*p != 1) {

            av_log(avctx, AV_LOG_ERROR, "Unknown avcC version %d\n", *p);

            return -1;

        }

        /* sps and pps in the avcC always have length coded with 2 bytes,

           so put a fake nal_length_size = 2 while parsing them */

        h->nal_length_size = 2;

        // Decode sps from avcC

        cnt = *(p+5) & 0x1f; // Number of sps

        p += 6;

        for (i = 0; i < cnt; i++) {

            nalsize = AV_RB16(p) + 2;

            if(decode_nal_units(h, p, nalsize) < 0) {

                av_log(avctx, AV_LOG_ERROR, "Decoding sps %d from avcC failed\n", i);

                return -1;

            }

            p += nalsize;

        }

        // Decode pps from avcC

        cnt = *(p++); // Number of pps

        for (i = 0; i < cnt; i++) {

            nalsize = AV_RB16(p) + 2;

            if(decode_nal_units(h, p, nalsize)  != nalsize) {

                av_log(avctx, AV_LOG_ERROR, "Decoding pps %d from avcC failed\n", i);

                return -1;

            }

            p += nalsize;

        }

        // Now store right nal length size, that will be use to parse all other nals

        h->nal_length_size = ((*(((char*)(avctx->extradata))+4))&0x03)+1;

        // Do not reparse avcC

        h->got_avcC = 1;

    }



    if(!h->got_avcC && !h->is_avc && s->avctx->extradata_size){

        if(decode_nal_units(h, s->avctx->extradata, s->avctx->extradata_size) < 0)

            return -1;

        h->got_avcC = 1;

    }



    buf_index=decode_nal_units(h, buf, buf_size);

    if(buf_index < 0)

        return -1;



    if(!(s->flags2 & CODEC_FLAG2_CHUNKS) && !s->current_picture_ptr){

        if (avctx->skip_frame >= AVDISCARD_NONREF || s->hurry_up) return 0;

        av_log(avctx, AV_LOG_ERROR, "no frame!\n");

        return -1;

    }



    if(!(s->flags2 & CODEC_FLAG2_CHUNKS) || (s->mb_y >= s->mb_height && s->mb_height)){

        Picture *out = s->current_picture_ptr;

        Picture *cur = s->current_picture_ptr;

        int i, pics, cross_idr, out_of_order, out_idx;



        field_end(h);



        if (cur->field_poc[0]==INT_MAX || cur->field_poc[1]==INT_MAX) {

            /* Wait for second field. */

            *data_size = 0;



        } else {

            cur->repeat_pict = 0;



            /* Signal interlacing information externally. */

            /* Prioritize picture timing SEI information over used decoding process if it exists. */

            if (h->sei_ct_type)

                cur->interlaced_frame = (h->sei_ct_type & (1<<1)) != 0;

            else

                cur->interlaced_frame = FIELD_OR_MBAFF_PICTURE;



            if(h->sps.pic_struct_present_flag){

                switch (h->sei_pic_struct)

                {

                case SEI_PIC_STRUCT_TOP_BOTTOM_TOP:

                case SEI_PIC_STRUCT_BOTTOM_TOP_BOTTOM:

                    // Signal the possibility of telecined film externally (pic_struct 5,6)

                    // From these hints, let the applications decide if they apply deinterlacing.

                    cur->repeat_pict = 1;

                    break;

                case SEI_PIC_STRUCT_FRAME_DOUBLING:

                    // Force progressive here, as doubling interlaced frame is a bad idea.

                    cur->interlaced_frame = 0;

                    cur->repeat_pict = 2;

                    break;

                case SEI_PIC_STRUCT_FRAME_TRIPLING:

                    cur->interlaced_frame = 0;

                    cur->repeat_pict = 4;

                    break;

                }

            }else{

                /* Derive interlacing flag from used decoding process. */

                cur->interlaced_frame = FIELD_OR_MBAFF_PICTURE;

            }



            if (cur->field_poc[0] != cur->field_poc[1]){

                /* Derive top_field_first from field pocs. */

                cur->top_field_first = cur->field_poc[0] < cur->field_poc[1];

            }else{

                if(cur->interlaced_frame || h->sps.pic_struct_present_flag){

                    /* Use picture timing SEI information. Even if it is a information of a past frame, better than nothing. */

                    if(h->sei_pic_struct == SEI_PIC_STRUCT_TOP_BOTTOM

                      || h->sei_pic_struct == SEI_PIC_STRUCT_TOP_BOTTOM_TOP)

                        cur->top_field_first = 1;

                    else

                        cur->top_field_first = 0;

                }else{

                    /* Most likely progressive */

                    cur->top_field_first = 0;

                }

            }



        //FIXME do something with unavailable reference frames



            /* Sort B-frames into display order */



            if(h->sps.bitstream_restriction_flag

               && s->avctx->has_b_frames < h->sps.num_reorder_frames){

                s->avctx->has_b_frames = h->sps.num_reorder_frames;

                s->low_delay = 0;

            }



            if(   s->avctx->strict_std_compliance >= FF_COMPLIANCE_STRICT

               && !h->sps.bitstream_restriction_flag){

                s->avctx->has_b_frames= MAX_DELAYED_PIC_COUNT;

                s->low_delay= 0;

            }



            pics = 0;

            while(h->delayed_pic[pics]) pics++;



            assert(pics <= MAX_DELAYED_PIC_COUNT);



            h->delayed_pic[pics++] = cur;

            if(cur->reference == 0)

                cur->reference = DELAYED_PIC_REF;



            out = h->delayed_pic[0];

            out_idx = 0;

            for(i=1; h->delayed_pic[i] && (h->delayed_pic[i]->poc && !h->delayed_pic[i]->key_frame); i++)

                if(h->delayed_pic[i]->poc < out->poc){

                    out = h->delayed_pic[i];

                    out_idx = i;

                }

            cross_idr = !h->delayed_pic[0]->poc || !!h->delayed_pic[i] || h->delayed_pic[0]->key_frame;



            out_of_order = !cross_idr && out->poc < h->outputed_poc;



            if(h->sps.bitstream_restriction_flag && s->avctx->has_b_frames >= h->sps.num_reorder_frames)

                { }

            else if((out_of_order && pics-1 == s->avctx->has_b_frames && s->avctx->has_b_frames < MAX_DELAYED_PIC_COUNT)

               || (s->low_delay &&

                ((!cross_idr && out->poc > h->outputed_poc + 2)

                 || cur->pict_type == FF_B_TYPE)))

            {

                s->low_delay = 0;

                s->avctx->has_b_frames++;

            }



            if(out_of_order || pics > s->avctx->has_b_frames){

                out->reference &= ~DELAYED_PIC_REF;

                for(i=out_idx; h->delayed_pic[i]; i++)

                    h->delayed_pic[i] = h->delayed_pic[i+1];

            }

            if(!out_of_order && pics > s->avctx->has_b_frames){

                *data_size = sizeof(AVFrame);



                h->outputed_poc = out->poc;

                *pict= *(AVFrame*)out;

            }else{

                av_log(avctx, AV_LOG_DEBUG, "no picture\n");

            }

        }

    }



    assert(pict->data[0] || !*data_size);

    ff_print_debug_info(s, pict);

//printf("out %d\n", (int)pict->data[0]);

#if 0 //?



    /* Return the Picture timestamp as the frame number */

    /* we subtract 1 because it is added on utils.c     */

    avctx->frame_number = s->picture_number - 1;

#endif

    return get_consumed_bytes(s, buf_index, buf_size);

}
