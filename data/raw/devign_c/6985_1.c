void ff_mpeg_draw_horiz_band(MpegEncContext *s, int y, int h)

{

    ff_draw_horiz_band(s->avctx, &s->current_picture.f,

                       &s->last_picture.f, y, h, s->picture_structure,

                       s->first_field, s->low_delay);

}
