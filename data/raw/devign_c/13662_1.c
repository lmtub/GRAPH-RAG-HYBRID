void ff_init_block_index(MpegEncContext *s){ //FIXME maybe rename

    const int linesize   = s->current_picture.f.linesize[0]; //not s->linesize as this would be wrong for field pics

    const int uvlinesize = s->current_picture.f.linesize[1];

    const int mb_size= 4;



    s->block_index[0]= s->b8_stride*(s->mb_y*2    ) - 2 + s->mb_x*2;

    s->block_index[1]= s->b8_stride*(s->mb_y*2    ) - 1 + s->mb_x*2;

    s->block_index[2]= s->b8_stride*(s->mb_y*2 + 1) - 2 + s->mb_x*2;

    s->block_index[3]= s->b8_stride*(s->mb_y*2 + 1) - 1 + s->mb_x*2;

    s->block_index[4]= s->mb_stride*(s->mb_y + 1)                + s->b8_stride*s->mb_height*2 + s->mb_x - 1;

    s->block_index[5]= s->mb_stride*(s->mb_y + s->mb_height + 2) + s->b8_stride*s->mb_height*2 + s->mb_x - 1;

    //block_index is not used by mpeg2, so it is not affected by chroma_format



    s->dest[0] = s->current_picture.f.data[0] + ((s->mb_x - 1) <<  mb_size);

    s->dest[1] = s->current_picture.f.data[1] + ((s->mb_x - 1) << (mb_size - s->chroma_x_shift));

    s->dest[2] = s->current_picture.f.data[2] + ((s->mb_x - 1) << (mb_size - s->chroma_x_shift));



    if(!(s->pict_type==AV_PICTURE_TYPE_B && s->avctx->draw_horiz_band && s->picture_structure==PICT_FRAME))

    {

        if(s->picture_structure==PICT_FRAME){

        s->dest[0] += s->mb_y *   linesize << mb_size;

        s->dest[1] += s->mb_y * uvlinesize << (mb_size - s->chroma_y_shift);

        s->dest[2] += s->mb_y * uvlinesize << (mb_size - s->chroma_y_shift);

        }else{

            s->dest[0] += (s->mb_y>>1) *   linesize << mb_size;

            s->dest[1] += (s->mb_y>>1) * uvlinesize << (mb_size - s->chroma_y_shift);

            s->dest[2] += (s->mb_y>>1) * uvlinesize << (mb_size - s->chroma_y_shift);

            assert((s->mb_y&1) == (s->picture_structure == PICT_BOTTOM_FIELD));

        }

    }

}
