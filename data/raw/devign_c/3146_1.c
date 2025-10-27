static int decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             uint8_t *buf, int buf_size)

{

    H264Context *h = avctx->priv_data;

    MpegEncContext *s = &h->s;

    AVFrame *pict = data;

    int buf_index;



    s->flags= avctx->flags;

    s->flags2= avctx->flags2;



   /* no supplementary picture */

    if (buf_size == 0) {

        return 0;

    }



    if(s->flags&CODEC_FLAG_TRUNCATED){

        int next= find_frame_end(h, buf, buf_size);



        if( ff_combine_frame(&s->parse_context, next, &buf, &buf_size) < 0 )

            return buf_size;

//printf("next:%d buf_size:%d last_index:%d\n", next, buf_size, s->parse_context.last_index);

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

            nalsize = BE_16(p) + 2;

            if(decode_nal_units(h, p, nalsize) < 0) {

                av_log(avctx, AV_LOG_ERROR, "Decoding sps %d from avcC failed\n", i);

                return -1;

            }

            p += nalsize;

        }

        // Decode pps from avcC

        cnt = *(p++); // Number of pps

        for (i = 0; i < cnt; i++) {

            nalsize = BE_16(p) + 2;

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



    if(!h->is_avc && s->avctx->extradata_size && s->picture_number==0){

        if(decode_nal_units(h, s->avctx->extradata, s->avctx->extradata_size) < 0)

            return -1;

    }



    buf_index=decode_nal_units(h, buf, buf_size);

    if(buf_index < 0)

        return -1;



    //FIXME do something with unavailable reference frames



//    if(ret==FRAME_SKIPPED) return get_consumed_bytes(s, buf_index, buf_size);

    if(!s->current_picture_ptr){

        av_log(h->s.avctx, AV_LOG_DEBUG, "error, NO frame\n");

        return -1;

    }



    {

        Picture *out = s->current_picture_ptr;

#if 0 //decode order

        *data_size = sizeof(AVFrame);

#else

        /* Sort B-frames into display order */

        Picture *cur = s->current_picture_ptr;

        Picture *prev = h->delayed_output_pic;

        int out_idx = 0;

        int pics = 0;

        int out_of_order;

        int cross_idr = 0;

        int dropped_frame = 0;

        int i;



        if(h->sps.bitstream_restriction_flag

           && s->avctx->has_b_frames < h->sps.num_reorder_frames){

            s->avctx->has_b_frames = h->sps.num_reorder_frames;

            s->low_delay = 0;

        }



        while(h->delayed_pic[pics]) pics++;

        h->delayed_pic[pics++] = cur;

        if(cur->reference == 0)

            cur->reference = 1;



        for(i=0; h->delayed_pic[i]; i++)

            if(h->delayed_pic[i]->key_frame || h->delayed_pic[i]->poc==0)

                cross_idr = 1;



        out = h->delayed_pic[0];

        for(i=1; h->delayed_pic[i] && !h->delayed_pic[i]->key_frame; i++)

            if(h->delayed_pic[i]->poc < out->poc){

                out = h->delayed_pic[i];

                out_idx = i;

            }



        out_of_order = !cross_idr && prev && out->poc < prev->poc;

        if(prev && pics <= s->avctx->has_b_frames)

            out = prev;

        else if((out_of_order && pics-1 == s->avctx->has_b_frames && pics < 15)

           || (s->low_delay &&

            ((!cross_idr && prev && out->poc > prev->poc + 2)

             || cur->pict_type == B_TYPE)))

        {

            s->low_delay = 0;

            s->avctx->has_b_frames++;

            out = prev;

        }

        else if(out_of_order)

            out = prev;



        if(out_of_order || pics > s->avctx->has_b_frames){

            dropped_frame = (out != h->delayed_pic[out_idx]);

            for(i=out_idx; h->delayed_pic[i]; i++)

                h->delayed_pic[i] = h->delayed_pic[i+1];

        }



        if(prev == out && !dropped_frame)

            *data_size = 0;

        else

            *data_size = sizeof(AVFrame);

        if(prev && prev != out && prev->reference == 1)

            prev->reference = 0;

        h->delayed_output_pic = out;

#endif



        if(out)

            *pict= *(AVFrame*)out;

        else

            av_log(avctx, AV_LOG_DEBUG, "no picture\n");

    }



    assert(pict->data[0] || !*data_size);

    ff_print_debug_info(s, pict);

//printf("out %d\n", (int)pict->data[0]);

#if 0 //?



    /* Return the Picture timestamp as the frame number */

    /* we substract 1 because it is added on utils.c    */

    avctx->frame_number = s->picture_number - 1;

#endif

    return get_consumed_bytes(s, buf_index, buf_size);

}
