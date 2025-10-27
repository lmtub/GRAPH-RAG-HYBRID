void ff_jpeg2000_set_significance(Jpeg2000T1Context *t1, int x, int y,

                                  int negative)

{

    x++;

    y++;

    t1->flags[y][x] |= JPEG2000_T1_SIG;

    if (negative) {

        t1->flags[y][x + 1] |= JPEG2000_T1_SIG_W | JPEG2000_T1_SGN_W;

        t1->flags[y][x - 1] |= JPEG2000_T1_SIG_E | JPEG2000_T1_SGN_E;

        t1->flags[y + 1][x] |= JPEG2000_T1_SIG_N | JPEG2000_T1_SGN_N;

        t1->flags[y - 1][x] |= JPEG2000_T1_SIG_S | JPEG2000_T1_SGN_S;

    } else {

        t1->flags[y][x + 1] |= JPEG2000_T1_SIG_W;

        t1->flags[y][x - 1] |= JPEG2000_T1_SIG_E;

        t1->flags[y + 1][x] |= JPEG2000_T1_SIG_N;

        t1->flags[y - 1][x] |= JPEG2000_T1_SIG_S;

    }

    t1->flags[y + 1][x + 1] |= JPEG2000_T1_SIG_NW;

    t1->flags[y + 1][x - 1] |= JPEG2000_T1_SIG_NE;

    t1->flags[y - 1][x + 1] |= JPEG2000_T1_SIG_SW;

    t1->flags[y - 1][x - 1] |= JPEG2000_T1_SIG_SE;

}
