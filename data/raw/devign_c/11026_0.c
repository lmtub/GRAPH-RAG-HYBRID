static void frame_start(H264Context *h){

    MpegEncContext * const s = &h->s;

    int i;



    MPV_frame_start(s, s->avctx);

    ff_er_frame_start(s);



    assert(s->linesize && s->uvlinesize);



    for(i=0; i<16; i++){

        h->block_offset[i]= 4*((scan8[i] - scan8[0])&7) + 4*s->linesize*((scan8[i] - scan8[0])>>3);

        h->block_offset[24+i]= 4*((scan8[i] - scan8[0])&7) + 8*s->linesize*((scan8[i] - scan8[0])>>3);

    }

    for(i=0; i<4; i++){

        h->block_offset[16+i]=

        h->block_offset[20+i]= 4*((scan8[i] - scan8[0])&7) + 4*s->uvlinesize*((scan8[i] - scan8[0])>>3);

        h->block_offset[24+16+i]=

        h->block_offset[24+20+i]= 4*((scan8[i] - scan8[0])&7) + 8*s->uvlinesize*((scan8[i] - scan8[0])>>3);

    }



    /* can't be in alloc_tables because linesize isn't known there.

     * FIXME: redo bipred weight to not require extra buffer? */

    if(!s->obmc_scratchpad)

        s->obmc_scratchpad = av_malloc(16*s->linesize + 2*8*s->uvlinesize);



//    s->decode= (s->flags&CODEC_FLAG_PSNR) || !s->encoding || s->current_picture.reference /*|| h->contains_intra*/ || 1;

}
