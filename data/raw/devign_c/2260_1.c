static void mpeg_er_decode_mb(void *opaque, int ref, int mv_dir, int mv_type,

                              int (*mv)[2][4][2],

                              int mb_x, int mb_y, int mb_intra, int mb_skipped)

{

    MpegEncContext *s = opaque;



    s->mv_dir     = mv_dir;

    s->mv_type    = mv_type;

    s->mb_intra   = mb_intra;

    s->mb_skipped = mb_skipped;

    s->mb_x       = mb_x;

    s->mb_y       = mb_y;

    memcpy(s->mv, mv, sizeof(*mv));



    ff_init_block_index(s);

    ff_update_block_index(s);



    s->dsp.clear_blocks(s->block[0]);



    s->dest[0] = s->current_picture.f.data[0] + (s->mb_y *  16                       * s->linesize)   + s->mb_x *  16;

    s->dest[1] = s->current_picture.f.data[1] + (s->mb_y * (16 >> s->chroma_y_shift) * s->uvlinesize) + s->mb_x * (16 >> s->chroma_x_shift);

    s->dest[2] = s->current_picture.f.data[2] + (s->mb_y * (16 >> s->chroma_y_shift) * s->uvlinesize) + s->mb_x * (16 >> s->chroma_x_shift);



    assert(ref == 0);

    ff_MPV_decode_mb(s, s->block);

}
