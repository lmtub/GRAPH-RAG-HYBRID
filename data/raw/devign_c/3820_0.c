static inline int vc1_i_pred_dc(MpegEncContext *s, int overlap, int pq, int n,

                              int16_t **dc_val_ptr, int *dir_ptr)

{

    int a, b, c, wrap, pred, scale;

    int16_t *dc_val;

    static const uint16_t dcpred[32] = {

    -1, 1024,  512,  341,  256,  205,  171,  146,  128,

         114,  102,   93,   85,   79,   73,   68,   64,

          60,   57,   54,   51,   49,   47,   45,   43,

          41,   39,   38,   37,   35,   34,   33

    };



    /* find prediction - wmv3_dc_scale always used here in fact */

    if (n < 4)     scale = s->y_dc_scale;

    else           scale = s->c_dc_scale;



    wrap = s->block_wrap[n];

    dc_val= s->dc_val[0] + s->block_index[n];



    /* B A

     * C X

     */

    c = dc_val[ - 1];

    b = dc_val[ - 1 - wrap];

    a = dc_val[ - wrap];



    if (pq < 9 || !overlap)

    {

        /* Set outer values */

        if (!s->mb_y && (n!=2 && n!=3)) b=a=dcpred[scale];

        if (s->mb_x == 0 && (n!=1 && n!=3)) b=c=dcpred[scale];

    }

    else

    {

        /* Set outer values */

        if (!s->mb_y && (n!=2 && n!=3)) b=a=0;

        if (s->mb_x == 0 && (n!=1 && n!=3)) b=c=0;

    }



    if (abs(a - b) <= abs(b - c)) {

        pred = c;

        *dir_ptr = 1;//left

    } else {

        pred = a;

        *dir_ptr = 0;//top

    }



    /* update predictor */

    *dc_val_ptr = &dc_val[0];

    return pred;

}
