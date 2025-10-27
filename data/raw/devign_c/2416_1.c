static void pred_spatial_direct_motion(H264Context * const h, int *mb_type){

    MpegEncContext * const s = &h->s;

    int b8_stride = 2;

    int b4_stride = h->b_stride;

    int mb_xy = h->mb_xy, mb_y = s->mb_y;

    int mb_type_col[2];

    const int16_t (*l1mv0)[2], (*l1mv1)[2];

    const int8_t *l1ref0, *l1ref1;

    const int is_b8x8 = IS_8X8(*mb_type);

    unsigned int sub_mb_type= MB_TYPE_L0L1;

    int i8, i4;

    int ref[2];

    int mv[2];

    int list;



    assert(h->ref_list[1][0].f.reference & 3);



    await_reference_mb_row(h, &h->ref_list[1][0], s->mb_y + !!IS_INTERLACED(*mb_type));



#define MB_TYPE_16x16_OR_INTRA (MB_TYPE_16x16|MB_TYPE_INTRA4x4|MB_TYPE_INTRA16x16|MB_TYPE_INTRA_PCM)





    /* ref = min(neighbors) */

    for(list=0; list<2; list++){

        int left_ref = h->ref_cache[list][scan8[0] - 1];

        int top_ref  = h->ref_cache[list][scan8[0] - 8];

        int refc = h->ref_cache[list][scan8[0] - 8 + 4];

        const int16_t *C= h->mv_cache[list][ scan8[0] - 8 + 4];

        if(refc == PART_NOT_AVAILABLE){

            refc = h->ref_cache[list][scan8[0] - 8 - 1];

            C    = h-> mv_cache[list][scan8[0] - 8 - 1];


        ref[list] = FFMIN3((unsigned)left_ref, (unsigned)top_ref, (unsigned)refc);

        if(ref[list] >= 0){

            //this is just pred_motion() but with the cases removed that cannot happen for direct blocks

            const int16_t * const A= h->mv_cache[list][ scan8[0] - 1 ];

            const int16_t * const B= h->mv_cache[list][ scan8[0] - 8 ];



            int match_count= (left_ref==ref[list]) + (top_ref==ref[list]) + (refc==ref[list]);

            if(match_count > 1){ //most common

                mv[list]= pack16to32(mid_pred(A[0], B[0], C[0]),

                                     mid_pred(A[1], B[1], C[1]) );

            }else {

                assert(match_count==1);

                if(left_ref==ref[list]){

                    mv[list]= AV_RN32A(A);

                }else if(top_ref==ref[list]){

                    mv[list]= AV_RN32A(B);

                }else{

                    mv[list]= AV_RN32A(C);



        }else{

            int mask= ~(MB_TYPE_L0 << (2*list));

            mv[list] = 0;

            ref[list] = -1;

            if(!is_b8x8)

                *mb_type &= mask;

            sub_mb_type &= mask;



    if(ref[0] < 0 && ref[1] < 0){

        ref[0] = ref[1] = 0;

        if(!is_b8x8)

            *mb_type |= MB_TYPE_L0L1;

        sub_mb_type |= MB_TYPE_L0L1;




    if(!(is_b8x8|mv[0]|mv[1])){

        fill_rectangle(&h->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);

        fill_rectangle(&h->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);

        fill_rectangle(&h->mv_cache[0][scan8[0]], 4, 4, 8, 0, 4);

        fill_rectangle(&h->mv_cache[1][scan8[0]], 4, 4, 8, 0, 4);

        *mb_type= (*mb_type & ~(MB_TYPE_8x8|MB_TYPE_16x8|MB_TYPE_8x16|MB_TYPE_P1L0|MB_TYPE_P1L1))|MB_TYPE_16x16|MB_TYPE_DIRECT2;

        return;




    if (IS_INTERLACED(h->ref_list[1][0].f.mb_type[mb_xy])) { // AFL/AFR/FR/FL -> AFL/FL

        if (!IS_INTERLACED(*mb_type)) {                          //     AFR/FR    -> AFL/FL

            mb_y = (s->mb_y&~1) + h->col_parity;

            mb_xy= s->mb_x + ((s->mb_y&~1) + h->col_parity)*s->mb_stride;

            b8_stride = 0;

        }else{

            mb_y  += h->col_fieldoff;

            mb_xy += s->mb_stride*h->col_fieldoff; // non zero for FL -> FL & differ parity


        goto single_col;

    }else{                                               // AFL/AFR/FR/FL -> AFR/FR

        if(IS_INTERLACED(*mb_type)){                     // AFL       /FL -> AFR/FR

            mb_y = s->mb_y&~1;

            mb_xy= s->mb_x + (s->mb_y&~1)*s->mb_stride;

            mb_type_col[0] = h->ref_list[1][0].f.mb_type[mb_xy];

            mb_type_col[1] = h->ref_list[1][0].f.mb_type[mb_xy + s->mb_stride];

            b8_stride = 2+4*s->mb_stride;

            b4_stride *= 6;







            sub_mb_type |= MB_TYPE_16x16|MB_TYPE_DIRECT2; /* B_SUB_8x8 */

            if(    (mb_type_col[0] & MB_TYPE_16x16_OR_INTRA)

                && (mb_type_col[1] & MB_TYPE_16x16_OR_INTRA)

                && !is_b8x8){

                *mb_type   |= MB_TYPE_16x8 |MB_TYPE_DIRECT2; /* B_16x8 */

            }else{

                *mb_type   |= MB_TYPE_8x8;


        }else{                                           //     AFR/FR    -> AFR/FR

single_col:

            mb_type_col[0] =

            mb_type_col[1] = h->ref_list[1][0].f.mb_type[mb_xy];



            sub_mb_type |= MB_TYPE_16x16|MB_TYPE_DIRECT2; /* B_SUB_8x8 */

            if(!is_b8x8 && (mb_type_col[0] & MB_TYPE_16x16_OR_INTRA)){

                *mb_type   |= MB_TYPE_16x16|MB_TYPE_DIRECT2; /* B_16x16 */

            }else if(!is_b8x8 && (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16))){

                *mb_type   |= MB_TYPE_DIRECT2 | (mb_type_col[0] & (MB_TYPE_16x8|MB_TYPE_8x16));

            }else{

                if(!h->sps.direct_8x8_inference_flag){

                    /* FIXME save sub mb types from previous frames (or derive from MVs)

                    * so we know exactly what block size to use */

                    sub_mb_type += (MB_TYPE_8x8-MB_TYPE_16x16); /* B_SUB_4x4 */


                *mb_type   |= MB_TYPE_8x8;






    await_reference_mb_row(h, &h->ref_list[1][0], mb_y);



    l1mv0  = &h->ref_list[1][0].f.motion_val[0][h->mb2b_xy [mb_xy]];

    l1mv1  = &h->ref_list[1][0].f.motion_val[1][h->mb2b_xy [mb_xy]];

    l1ref0 = &h->ref_list[1][0].f.ref_index [0][4 * mb_xy];

    l1ref1 = &h->ref_list[1][0].f.ref_index [1][4 * mb_xy];

    if(!b8_stride){

        if(s->mb_y&1){

            l1ref0 += 2;

            l1ref1 += 2;

            l1mv0  +=  2*b4_stride;

            l1mv1  +=  2*b4_stride;







        if(IS_INTERLACED(*mb_type) != IS_INTERLACED(mb_type_col[0])){

            int n=0;

            for(i8=0; i8<4; i8++){

                int x8 = i8&1;

                int y8 = i8>>1;

                int xy8 = x8+y8*b8_stride;

                int xy4 = 3*x8+y8*b4_stride;

                int a,b;



                if(is_b8x8 && !IS_DIRECT(h->sub_mb_type[i8]))

                    continue;

                h->sub_mb_type[i8] = sub_mb_type;



                fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[0], 1);

                fill_rectangle(&h->ref_cache[1][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[1], 1);

                if(!IS_INTRA(mb_type_col[y8]) && !h->ref_list[1][0].long_ref

                   && (   (l1ref0[xy8] == 0 && FFABS(l1mv0[xy4][0]) <= 1 && FFABS(l1mv0[xy4][1]) <= 1)

                       || (l1ref0[xy8]  < 0 && l1ref1[xy8] == 0 && FFABS(l1mv1[xy4][0]) <= 1 && FFABS(l1mv1[xy4][1]) <= 1))){

                    a=b=0;

                    if(ref[0] > 0)

                        a= mv[0];

                    if(ref[1] > 0)

                        b= mv[1];

                    n++;

                }else{

                    a= mv[0];

                    b= mv[1];


                fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, a, 4);

                fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, b, 4);


            if(!is_b8x8 && !(n&3))

                *mb_type= (*mb_type & ~(MB_TYPE_8x8|MB_TYPE_16x8|MB_TYPE_8x16|MB_TYPE_P1L0|MB_TYPE_P1L1))|MB_TYPE_16x16|MB_TYPE_DIRECT2;

        }else if(IS_16X16(*mb_type)){

            int a,b;



            fill_rectangle(&h->ref_cache[0][scan8[0]], 4, 4, 8, (uint8_t)ref[0], 1);

            fill_rectangle(&h->ref_cache[1][scan8[0]], 4, 4, 8, (uint8_t)ref[1], 1);

            if(!IS_INTRA(mb_type_col[0]) && !h->ref_list[1][0].long_ref

               && (   (l1ref0[0] == 0 && FFABS(l1mv0[0][0]) <= 1 && FFABS(l1mv0[0][1]) <= 1)

                   || (l1ref0[0]  < 0 && l1ref1[0] == 0 && FFABS(l1mv1[0][0]) <= 1 && FFABS(l1mv1[0][1]) <= 1

                       && h->x264_build>33U))){

                a=b=0;

                if(ref[0] > 0)

                    a= mv[0];

                if(ref[1] > 0)

                    b= mv[1];

            }else{

                a= mv[0];

                b= mv[1];


            fill_rectangle(&h->mv_cache[0][scan8[0]], 4, 4, 8, a, 4);

            fill_rectangle(&h->mv_cache[1][scan8[0]], 4, 4, 8, b, 4);

        }else{

            int n=0;

            for(i8=0; i8<4; i8++){

                const int x8 = i8&1;

                const int y8 = i8>>1;



                if(is_b8x8 && !IS_DIRECT(h->sub_mb_type[i8]))

                    continue;

                h->sub_mb_type[i8] = sub_mb_type;



                fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, mv[0], 4);

                fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, mv[1], 4);

                fill_rectangle(&h->ref_cache[0][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[0], 1);

                fill_rectangle(&h->ref_cache[1][scan8[i8*4]], 2, 2, 8, (uint8_t)ref[1], 1);



                assert(b8_stride==2);

                /* col_zero_flag */

                if(!IS_INTRA(mb_type_col[0]) && !h->ref_list[1][0].long_ref && (   l1ref0[i8] == 0

                                              || (l1ref0[i8] < 0 && l1ref1[i8] == 0

                                                  && h->x264_build>33U))){

                    const int16_t (*l1mv)[2]= l1ref0[i8] == 0 ? l1mv0 : l1mv1;

                    if(IS_SUB_8X8(sub_mb_type)){

                        const int16_t *mv_col = l1mv[x8*3 + y8*3*b4_stride];

                        if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){

                            if(ref[0] == 0)

                                fill_rectangle(&h->mv_cache[0][scan8[i8*4]], 2, 2, 8, 0, 4);

                            if(ref[1] == 0)

                                fill_rectangle(&h->mv_cache[1][scan8[i8*4]], 2, 2, 8, 0, 4);

                            n+=4;


                    }else{

                        int m=0;

                    for(i4=0; i4<4; i4++){

                        const int16_t *mv_col = l1mv[x8*2 + (i4&1) + (y8*2 + (i4>>1))*b4_stride];

                        if(FFABS(mv_col[0]) <= 1 && FFABS(mv_col[1]) <= 1){

                            if(ref[0] == 0)

                                AV_ZERO32(h->mv_cache[0][scan8[i8*4+i4]]);

                            if(ref[1] == 0)

                                AV_ZERO32(h->mv_cache[1][scan8[i8*4+i4]]);

                            m++;



                    if(!(m&3))

                        h->sub_mb_type[i8]+= MB_TYPE_16x16 - MB_TYPE_8x8;

                    n+=m;




            if(!is_b8x8 && !(n&15))

                *mb_type= (*mb_type & ~(MB_TYPE_8x8|MB_TYPE_16x8|MB_TYPE_8x16|MB_TYPE_P1L0|MB_TYPE_P1L1))|MB_TYPE_16x16|MB_TYPE_DIRECT2;

