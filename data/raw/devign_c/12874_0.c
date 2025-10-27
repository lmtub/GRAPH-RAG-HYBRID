static void encode_sigpass(Jpeg2000T1Context *t1, int width, int height, int bandno, int *nmsedec, int bpno)

{

    int y0, x, y, mask = 1 << (bpno + NMSEDEC_FRACBITS);

    for (y0 = 0; y0 < height; y0 += 4)

        for (x = 0; x < width; x++)

            for (y = y0; y < height && y < y0+4; y++){

                if (!(t1->flags[y+1][x+1] & JPEG2000_T1_SIG) && (t1->flags[y+1][x+1] & JPEG2000_T1_SIG_NB)){

                    int ctxno = ff_jpeg2000_getsigctxno(t1->flags[y+1][x+1], bandno),

                        bit = t1->data[y][x] & mask ? 1 : 0;

                    ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, bit);

                    if (bit){

                        int xorbit;

                        int ctxno = ff_jpeg2000_getsgnctxno(t1->flags[y+1][x+1], &xorbit);

                        ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, (t1->flags[y+1][x+1] >> 15) ^ xorbit);

                        *nmsedec += getnmsedec_sig(t1->data[y][x], bpno + NMSEDEC_FRACBITS);

                        ff_jpeg2000_set_significance(t1, x, y, t1->flags[y+1][x+1] >> 15);

                    }

                    t1->flags[y+1][x+1] |= JPEG2000_T1_VIS;

                }

            }

}
