static void encode_refpass(Jpeg2000T1Context *t1, int width, int height, int *nmsedec, int bpno)

{

    int y0, x, y, mask = 1 << (bpno + NMSEDEC_FRACBITS);

    for (y0 = 0; y0 < height; y0 += 4)

        for (x = 0; x < width; x++)

            for (y = y0; y < height && y < y0+4; y++)

                if ((t1->flags[y+1][x+1] & (JPEG2000_T1_SIG | JPEG2000_T1_VIS)) == JPEG2000_T1_SIG){

                    int ctxno = ff_jpeg2000_getrefctxno(t1->flags[y+1][x+1]);

                    *nmsedec += getnmsedec_ref(t1->data[y][x], bpno + NMSEDEC_FRACBITS);

                    ff_mqc_encode(&t1->mqc, t1->mqc.cx_states + ctxno, t1->data[y][x] & mask ? 1:0);

                    t1->flags[y+1][x+1] |= JPEG2000_T1_REF;

                }

}
