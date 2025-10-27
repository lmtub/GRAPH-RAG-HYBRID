static int sse_mb(MpegEncContext *s){

    int w= 16;

    int h= 16;



    if(s->mb_x*16 + 16 > s->width ) w= s->width - s->mb_x*16;

    if(s->mb_y*16 + 16 > s->height) h= s->height- s->mb_y*16;



    if(w==16 && h==16)

      if(s->avctx->mb_cmp == FF_CMP_NSSE){

        return  s->dsp.nsse[0](s, s->new_picture.f.data[0] + s->mb_x*16 + s->mb_y*s->linesize*16, s->dest[0], s->linesize, 16)

               +s->dsp.nsse[1](s, s->new_picture.f.data[1] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,s->dest[1], s->uvlinesize, 8)

               +s->dsp.nsse[1](s, s->new_picture.f.data[2] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,s->dest[2], s->uvlinesize, 8);

      }else{

        return  s->dsp.sse[0](NULL, s->new_picture.f.data[0] + s->mb_x*16 + s->mb_y*s->linesize*16, s->dest[0], s->linesize, 16)

               +s->dsp.sse[1](NULL, s->new_picture.f.data[1] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,s->dest[1], s->uvlinesize, 8)

               +s->dsp.sse[1](NULL, s->new_picture.f.data[2] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,s->dest[2], s->uvlinesize, 8);

      }

    else

        return  sse(s, s->new_picture.f.data[0] + s->mb_x*16 + s->mb_y*s->linesize*16, s->dest[0], w, h, s->linesize)

               +sse(s, s->new_picture.f.data[1] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,s->dest[1], w>>1, h>>1, s->uvlinesize)

               +sse(s, s->new_picture.f.data[2] + s->mb_x*8  + s->mb_y*s->uvlinesize*8,s->dest[2], w>>1, h>>1, s->uvlinesize);

}
