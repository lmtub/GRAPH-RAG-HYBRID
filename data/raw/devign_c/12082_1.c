static void dequantization_float(int x, int y, Jpeg2000Cblk *cblk,

                                 Jpeg2000Component *comp,

                                 Jpeg2000T1Context *t1, Jpeg2000Band *band)

{

    int i, j, idx;

    float *datap = &comp->data[(comp->coord[0][1] - comp->coord[0][0]) * y + x];

    for (j = 0; j < (cblk->coord[1][1] - cblk->coord[1][0]); ++j)

        for (i = 0; i < (cblk->coord[0][1] - cblk->coord[0][0]); ++i) {

            idx        = (comp->coord[0][1] - comp->coord[0][0]) * j + i;

            datap[idx] = (float)(t1->data[j][i]) * band->f_stepsize;

        }

}
