static void vc1_decode_skip_blocks(VC1Context *v)

{

    MpegEncContext *s = &v->s;



    if (!v->s.last_picture.f.data[0])

        return;



    ff_er_add_slice(&s->er, 0, s->start_mb_y, s->mb_width - 1, s->end_mb_y - 1, ER_MB_END);

    s->first_slice_line = 1;

    for (s->mb_y = s->start_mb_y; s->mb_y < s->end_mb_y; s->mb_y++) {

        s->mb_x = 0;

        init_block_index(v);

        ff_update_block_index(s);

        memcpy(s->dest[0], s->last_picture.f.data[0] + s->mb_y * 16 * s->linesize,   s->linesize   * 16);

        memcpy(s->dest[1], s->last_picture.f.data[1] + s->mb_y *  8 * s->uvlinesize, s->uvlinesize *  8);

        memcpy(s->dest[2], s->last_picture.f.data[2] + s->mb_y *  8 * s->uvlinesize, s->uvlinesize *  8);

        ff_mpeg_draw_horiz_band(s, s->mb_y * 16, 16);

        s->first_slice_line = 0;

    }

    s->pict_type = AV_PICTURE_TYPE_P;

}
