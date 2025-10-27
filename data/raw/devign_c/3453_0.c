static void predict_plane(SnowContext *s, DWTELEM *buf, int plane_index, int add){

    Plane *p= &s->plane[plane_index];

    const int mb_w= s->mb_band.width;

    const int mb_h= s->mb_band.height;

    const int mb_stride= s->mb_band.stride;

    int x, y, mb_x, mb_y;

    int scale      = plane_index ?  s->mv_scale : 2*s->mv_scale;

    int block_w    = plane_index ?  8 : 16;

    uint8_t *obmc  = plane_index ? obmc16 : obmc32;

    int obmc_stride= plane_index ? 16 : 32;

    int ref_stride= s->last_picture.linesize[plane_index];

    uint8_t *ref  = s->last_picture.data[plane_index];

    int w= p->width;

    int h= p->height;

    

if(s->avctx->debug&512){

    for(y=0; y<h; y++){

        for(x=0; x<w; x++){

            if(add) buf[x + y*w]+= 128*256;

            else    buf[x + y*w]-= 128*256;

        }

    }

    

    return;

}

    for(mb_y=-1; mb_y<=mb_h; mb_y++){

        for(mb_x=-1; mb_x<=mb_w; mb_x++){

            int index= clip(mb_x, 0, mb_w-1) + clip(mb_y, 0, mb_h-1)*mb_stride;



            add_xblock(buf, ref, obmc, 

                       block_w*mb_x - block_w/2, 

                       block_w*mb_y - block_w/2,

                       2*block_w, 2*block_w,

                       s->mv_band[0].buf[index]*scale, s->mv_band[1].buf[index]*scale,

                       w, h,

                       w, ref_stride, obmc_stride, 

                       s->mb_band.buf[index], add);



        }

    }

}
