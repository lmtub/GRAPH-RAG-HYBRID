static void decode_refpass(Jpeg2000T1Context *t1, int width, int height,

                           int bpno, int vert_causal_ctx_csty_symbol)

{

    int phalf, nhalf;

    int y0, x, y;



    phalf = 1 << (bpno - 1);

    nhalf = -phalf;



    for (y0 = 0; y0 < height; y0 += 4)

        for (x = 0; x < width; x++)

            for (y = y0; y < height && y < y0 + 4; y++)

                if ((t1->flags[y + 1][x + 1] & (JPEG2000_T1_SIG | JPEG2000_T1_VIS)) == JPEG2000_T1_SIG) {

                    int flags_mask = (vert_causal_ctx_csty_symbol && y == y0 + 3) ?

                        ~(JPEG2000_T1_SIG_S | JPEG2000_T1_SIG_SW | JPEG2000_T1_SIG_SE | JPEG2000_T1_SGN_S) : -1;

                    int ctxno = ff_jpeg2000_getrefctxno(t1->flags[y + 1][x + 1] & flags_mask);

                    int r     = ff_mqc_decode(&t1->mqc,

                                              t1->mqc.cx_states + ctxno)

                                ? phalf : nhalf;

                    t1->data[y][x]          += t1->data[y][x] < 0 ? -r : r;

                    t1->flags[y + 1][x + 1] |= JPEG2000_T1_REF;

                }

}
