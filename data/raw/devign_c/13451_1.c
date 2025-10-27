int ff_MPV_lowest_referenced_row(MpegEncContext *s, int dir)

{

    int my_max = INT_MIN, my_min = INT_MAX, qpel_shift = !s->quarter_sample;

    int my, off, i, mvs;



    if (s->picture_structure != PICT_FRAME) goto unhandled;



    switch (s->mv_type) {

        case MV_TYPE_16X16:

            mvs = 1;

            break;

        case MV_TYPE_16X8:

            mvs = 2;

            break;

        case MV_TYPE_8X8:

            mvs = 4;

            break;

        default:

            goto unhandled;

    }



    for (i = 0; i < mvs; i++) {

        my = s->mv[dir][i][1]<<qpel_shift;

        my_max = FFMAX(my_max, my);

        my_min = FFMIN(my_min, my);

    }



    off = (FFMAX(-my_min, my_max) + 63) >> 6;



    return FFMIN(FFMAX(s->mb_y + off, 0), s->mb_height-1);

unhandled:

    return s->mb_height-1;

}
