static void loop_filter(H264Context *h, int start_x, int end_x){

    MpegEncContext * const s = &h->s;

    uint8_t  *dest_y, *dest_cb, *dest_cr;

    int linesize, uvlinesize, mb_x, mb_y;

    const int end_mb_y= s->mb_y + FRAME_MBAFF;

    const int old_slice_type= h->slice_type;

    const int pixel_shift = h->pixel_shift;



    if(h->deblocking_filter) {

        for(mb_x= start_x; mb_x<end_x; mb_x++){

            for(mb_y=end_mb_y - FRAME_MBAFF; mb_y<= end_mb_y; mb_y++){

                int mb_xy, mb_type;

                mb_xy = h->mb_xy = mb_x + mb_y*s->mb_stride;

                h->slice_num= h->slice_table[mb_xy];

                mb_type= s->current_picture.mb_type[mb_xy];

                h->list_count= h->list_counts[mb_xy];



                if(FRAME_MBAFF)

                    h->mb_mbaff = h->mb_field_decoding_flag = !!IS_INTERLACED(mb_type);



                s->mb_x= mb_x;

                s->mb_y= mb_y;

                dest_y  = s->current_picture.data[0] + ((mb_x << pixel_shift) + mb_y * s->linesize  ) * 16;

                dest_cb = s->current_picture.data[1] + ((mb_x << pixel_shift) + mb_y * s->uvlinesize) * 8;

                dest_cr = s->current_picture.data[2] + ((mb_x << pixel_shift) + mb_y * s->uvlinesize) * 8;

                    //FIXME simplify above



                if (MB_FIELD) {

                    linesize   = h->mb_linesize   = s->linesize * 2;

                    uvlinesize = h->mb_uvlinesize = s->uvlinesize * 2;

                    if(mb_y&1){ //FIXME move out of this function?

                        dest_y -= s->linesize*15;

                        dest_cb-= s->uvlinesize*7;

                        dest_cr-= s->uvlinesize*7;

                    }

                } else {

                    linesize   = h->mb_linesize   = s->linesize;

                    uvlinesize = h->mb_uvlinesize = s->uvlinesize;

                }

                backup_mb_border(h, dest_y, dest_cb, dest_cr, linesize, uvlinesize, 0);

                if(fill_filter_caches(h, mb_type))

                    continue;

                h->chroma_qp[0] = get_chroma_qp(h, 0, s->current_picture.qscale_table[mb_xy]);

                h->chroma_qp[1] = get_chroma_qp(h, 1, s->current_picture.qscale_table[mb_xy]);



                if (FRAME_MBAFF) {

                    ff_h264_filter_mb     (h, mb_x, mb_y, dest_y, dest_cb, dest_cr, linesize, uvlinesize);

                } else {

                    ff_h264_filter_mb_fast(h, mb_x, mb_y, dest_y, dest_cb, dest_cr, linesize, uvlinesize);

                }

            }

        }

    }

    h->slice_type= old_slice_type;

    s->mb_x= 0;

    s->mb_y= end_mb_y - FRAME_MBAFF;

    h->chroma_qp[0] = get_chroma_qp(h, 0, s->qscale);

    h->chroma_qp[1] = get_chroma_qp(h, 1, s->qscale);

}
