static int epzs_motion_search(MpegEncContext * s,

                             int *mx_ptr, int *my_ptr,

                             int P[5][2], int pred_x, int pred_y,

                             int xmin, int ymin, int xmax, int ymax)

{

    int best[2]={0, 0};

    int d, dmin; 

    UINT8 *new_pic, *old_pic;

    const int pic_stride= s->linesize;

    const int pic_xy= (s->mb_y*pic_stride + s->mb_x)*16;

    UINT16 *mv_penalty= s->mv_penalty[s->f_code] + MAX_MV; // f_code of the prev frame

    int quant= s->qscale; // qscale of the prev frame

    const int shift= 1+s->quarter_sample;



    new_pic = s->new_picture[0] + pic_xy;

    old_pic = s->last_picture[0] + pic_xy;

//printf("%d %d %d %d\n", xmin, ymin, xmax, ymax);

    

    dmin = pix_abs16x16(new_pic, old_pic, pic_stride, 16);

    if(dmin<Z_THRESHOLD){

        *mx_ptr= 0;

        *my_ptr= 0;

//printf("Z");

        return dmin;

    }



    /* first line */

    if ((s->mb_y == 0 || s->first_slice_line || s->first_gob_line)) {

        CHECK_MV(P[1][0]>>shift, P[1][1]>>shift)

    }else{

        CHECK_MV(P[4][0]>>shift, P[4][1]>>shift)

        if(dmin<Z_THRESHOLD){

            *mx_ptr= P[4][0]>>shift;

            *my_ptr= P[4][1]>>shift;

//printf("M\n");

            return dmin;

        }

        CHECK_MV(P[1][0]>>shift, P[1][1]>>shift)

        CHECK_MV(P[2][0]>>shift, P[2][1]>>shift)

        CHECK_MV(P[3][0]>>shift, P[3][1]>>shift)

    }

    CHECK_MV(P[0][0]>>shift, P[0][1]>>shift)



    dmin= small_diamond_search(s, best, dmin, new_pic, old_pic, pic_stride, 

                               pred_x, pred_y, mv_penalty, quant, xmin, ymin, xmax, ymax, shift);

    *mx_ptr= best[0];

    *my_ptr= best[1];    



//    printf("%d %d %d \n", best[0], best[1], dmin);



    return dmin;

}
