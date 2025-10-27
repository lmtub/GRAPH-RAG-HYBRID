static inline void mc_dir_part(H264Context *h, Picture *pic, int n, int square, int chroma_height, int delta, int list,

                           uint8_t *dest_y, uint8_t *dest_cb, uint8_t *dest_cr,

                           int src_x_offset, int src_y_offset,

                           qpel_mc_func *qpix_op, h264_chroma_mc_func chroma_op){

    MpegEncContext * const s = &h->s;

    const int mx= h->mv_cache[list][ scan8[n] ][0] + src_x_offset*8;

    const int my= h->mv_cache[list][ scan8[n] ][1] + src_y_offset*8;

    const int luma_xy= (mx&3) + ((my&3)<<2);

    uint8_t * src_y = pic->data[0] + (mx>>2) + (my>>2)*s->linesize;

    uint8_t * src_cb= pic->data[1] + (mx>>3) + (my>>3)*s->uvlinesize;

    uint8_t * src_cr= pic->data[2] + (mx>>3) + (my>>3)*s->uvlinesize;

    int extra_width= (s->flags&CODEC_FLAG_EMU_EDGE) ? 0 : 16; //FIXME increase edge?, IMHO not worth it

    int extra_height= extra_width;

    int emu=0;

    const int full_mx= mx>>2;

    const int full_my= my>>2;

    const int pic_width  = 16*s->mb_width;

    const int pic_height = 16*s->mb_height;



    assert(pic->data[0]);



    if(mx&7) extra_width -= 3;

    if(my&7) extra_height -= 3;



    if(   full_mx < 0-extra_width

       || full_my < 0-extra_height

       || full_mx + 16/*FIXME*/ > pic_width + extra_width

       || full_my + 16/*FIXME*/ > pic_height + extra_height){

        ff_emulated_edge_mc(s->edge_emu_buffer, src_y - 2 - 2*s->linesize, s->linesize, 16+5, 16+5/*FIXME*/, full_mx-2, full_my-2, pic_width, pic_height);

            src_y= s->edge_emu_buffer + 2 + 2*s->linesize;

        emu=1;

    }



    qpix_op[luma_xy](dest_y, src_y, s->linesize); //FIXME try variable height perhaps?

    if(!square){

        qpix_op[luma_xy](dest_y + delta, src_y + delta, s->linesize);

    }



    if(s->flags&CODEC_FLAG_GRAY) return;



    if(emu){

        ff_emulated_edge_mc(s->edge_emu_buffer, src_cb, s->uvlinesize, 9, 9/*FIXME*/, (mx>>3), (my>>3), pic_width>>1, pic_height>>1);

            src_cb= s->edge_emu_buffer;

    }

    chroma_op(dest_cb, src_cb, s->uvlinesize, chroma_height, mx&7, my&7);



    if(emu){

        ff_emulated_edge_mc(s->edge_emu_buffer, src_cr, s->uvlinesize, 9, 9/*FIXME*/, (mx>>3), (my>>3), pic_width>>1, pic_height>>1);

            src_cr= s->edge_emu_buffer;

    }

    chroma_op(dest_cr, src_cr, s->uvlinesize, chroma_height, mx&7, my&7);

}
