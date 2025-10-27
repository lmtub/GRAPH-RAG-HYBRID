static av_always_inline void predict_slice_buffered(SnowContext *s, slice_buffer * sb, IDWTELEM * old_buffer, int plane_index, int add, int mb_y){

    Plane *p= &s->plane[plane_index];

    const int mb_w= s->b_width  << s->block_max_depth;

    const int mb_h= s->b_height << s->block_max_depth;

    int x, y, mb_x;

    int block_size = MB_SIZE >> s->block_max_depth;

    int block_w    = plane_index ? block_size>>s->chroma_h_shift : block_size;

    int block_h    = plane_index ? block_size>>s->chroma_v_shift : block_size;

    const uint8_t *obmc  = plane_index ? ff_obmc_tab[s->block_max_depth+s->chroma_h_shift] : ff_obmc_tab[s->block_max_depth];

    int obmc_stride= plane_index ? (2*block_size)>>s->chroma_h_shift : 2*block_size;

    int ref_stride= s->current_picture->linesize[plane_index];

    uint8_t *dst8= s->current_picture->data[plane_index];

    int w= p->width;

    int h= p->height;



    if(s->keyframe || (s->avctx->debug&512)){

        if(mb_y==mb_h)

            return;



        if(add){

            for(y=block_h*mb_y; y<FFMIN(h,block_h*(mb_y+1)); y++){

//                DWTELEM * line = slice_buffer_get_line(sb, y);

                IDWTELEM * line = sb->line[y];

                for(x=0; x<w; x++){

//                    int v= buf[x + y*w] + (128<<FRAC_BITS) + (1<<(FRAC_BITS-1));

                    int v= line[x] + (128<<FRAC_BITS) + (1<<(FRAC_BITS-1));

                    v >>= FRAC_BITS;

                    if(v&(~255)) v= ~(v>>31);

                    dst8[x + y*ref_stride]= v;

                }

            }

        }else{

            for(y=block_h*mb_y; y<FFMIN(h,block_h*(mb_y+1)); y++){

//                DWTELEM * line = slice_buffer_get_line(sb, y);

                IDWTELEM * line = sb->line[y];

                for(x=0; x<w; x++){

                    line[x] -= 128 << FRAC_BITS;

//                    buf[x + y*w]-= 128<<FRAC_BITS;

                }

            }

        }



        return;

    }



    for(mb_x=0; mb_x<=mb_w; mb_x++){

        add_yblock(s, 1, sb, old_buffer, dst8, obmc,

                   block_w*mb_x - block_w/2,

                   block_h*mb_y - block_h/2,

                   block_w, block_h,

                   w, h,

                   w, ref_stride, obmc_stride,

                   mb_x - 1, mb_y - 1,

                   add, 0, plane_index);

    }



    if(s->avmv && mb_y < mb_h && plane_index == 0)

        for(mb_x=0; mb_x<mb_w; mb_x++){

            AVMotionVector *avmv = s->avmv + (s->avmv_index++);

            const int b_width = s->b_width  << s->block_max_depth;

            const int b_stride= b_width;

            BlockNode *bn= &s->block[mb_x + mb_y*b_stride];



            if (bn->type)

                continue;



            avmv->w = block_w;

            avmv->h = block_h;

            avmv->dst_x = block_w*mb_x - block_w/2;

            avmv->dst_y = block_h*mb_y - block_h/2;

            avmv->src_x = avmv->dst_x + (bn->mx * s->mv_scale)/8;

            avmv->src_y = avmv->dst_y + (bn->my * s->mv_scale)/8;

            avmv->source= -1 - bn->ref;

            avmv->flags = 0;

        }

}
