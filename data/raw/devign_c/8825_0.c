void ff_estimate_p_frame_motion(MpegEncContext * s,

                                int mb_x, int mb_y)

{

    UINT8 *pix, *ppix;

    int sum, varc, vard, mx, my, range, dmin, xx, yy;

    int xmin, ymin, xmax, ymax;

    int rel_xmin, rel_ymin, rel_xmax, rel_ymax;

    int pred_x=0, pred_y=0;

    int P[6][2];

    const int shift= 1+s->quarter_sample;

    int mb_type=0;

    uint8_t *ref_picture= s->last_picture[0];



    get_limits(s, &range, &xmin, &ymin, &xmax, &ymax, s->f_code);



    switch(s->me_method) {

    case ME_ZERO:

    default:

	no_motion_search(s, &mx, &my);

        dmin = 0;

        break;

    case ME_FULL:

	dmin = full_motion_search(s, &mx, &my, range, xmin, ymin, xmax, ymax, ref_picture);

        break;

    case ME_LOG:

	dmin = log_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax, ref_picture);

        break;

    case ME_PHODS:

	dmin = phods_motion_search(s, &mx, &my, range / 2, xmin, ymin, xmax, ymax, ref_picture);

        break;

    case ME_X1:

    case ME_EPZS:

       {

            const int mot_stride = s->block_wrap[0];

            const int mot_xy = s->block_index[0];



            rel_xmin= xmin - mb_x*16;

            rel_xmax= xmax - mb_x*16;

            rel_ymin= ymin - mb_y*16;

            rel_ymax= ymax - mb_y*16;



            P[0][0] = s->motion_val[mot_xy    ][0];

            P[0][1] = s->motion_val[mot_xy    ][1];

            P[1][0] = s->motion_val[mot_xy - 1][0];

            P[1][1] = s->motion_val[mot_xy - 1][1];

            if(P[1][0] > (rel_xmax<<shift)) P[1][0]= (rel_xmax<<shift);



            /* special case for first line */

            if ((mb_y == 0 || s->first_slice_line || s->first_gob_line)) {

                P[4][0] = P[1][0];

                P[4][1] = P[1][1];

            } else {

                P[2][0] = s->motion_val[mot_xy - mot_stride             ][0];

                P[2][1] = s->motion_val[mot_xy - mot_stride             ][1];

                P[3][0] = s->motion_val[mot_xy - mot_stride + 2         ][0];

                P[3][1] = s->motion_val[mot_xy - mot_stride + 2         ][1];

                if(P[2][1] > (rel_ymax<<shift)) P[2][1]= (rel_ymax<<shift);

                if(P[3][0] < (rel_xmin<<shift)) P[3][0]= (rel_xmin<<shift);

                if(P[3][1] > (rel_ymax<<shift)) P[3][1]= (rel_ymax<<shift);

        

                P[4][0]= mid_pred(P[1][0], P[2][0], P[3][0]);

                P[4][1]= mid_pred(P[1][1], P[2][1], P[3][1]);

            }

            if(s->out_format == FMT_H263){

                pred_x = P[4][0];

                pred_y = P[4][1];

            }else { /* mpeg1 at least */

                pred_x= P[1][0];

                pred_y= P[1][1];

            }

        }

        dmin = epzs_motion_search(s, &mx, &my, P, pred_x, pred_y, rel_xmin, rel_ymin, rel_xmax, rel_ymax, ref_picture);

 

        mx+= mb_x*16;

        my+= mb_y*16;

        break;

    }

    

    if(s->flags&CODEC_FLAG_4MV){

        int block;



        mb_type|= MB_TYPE_INTER4V;



        for(block=0; block<4; block++){

            int mx4, my4;

            int pred_x4, pred_y4;

            int dmin4;

            static const int off[4]= {2, 1, 1, -1};

            const int mot_stride = s->block_wrap[0];

            const int mot_xy = s->block_index[block];

            const int block_x= mb_x*2 + (block&1);

            const int block_y= mb_y*2 + (block>>1);



            const int rel_xmin4= xmin - block_x*8;

            const int rel_xmax4= xmax - block_x*8 + 8;

            const int rel_ymin4= ymin - block_y*8;

            const int rel_ymax4= ymax - block_y*8 + 8;



            P[0][0] = s->motion_val[mot_xy    ][0];

            P[0][1] = s->motion_val[mot_xy    ][1];

            P[1][0] = s->motion_val[mot_xy - 1][0];

            P[1][1] = s->motion_val[mot_xy - 1][1];

            if(P[1][0] > (rel_xmax4<<shift)) P[1][0]= (rel_xmax4<<shift);



            /* special case for first line */

            if ((mb_y == 0 || s->first_slice_line || s->first_gob_line) && block<2) {

                P[4][0] = P[1][0];

                P[4][1] = P[1][1];

            } else {

                P[2][0] = s->motion_val[mot_xy - mot_stride             ][0];

                P[2][1] = s->motion_val[mot_xy - mot_stride             ][1];

                P[3][0] = s->motion_val[mot_xy - mot_stride + off[block]][0];

                P[3][1] = s->motion_val[mot_xy - mot_stride + off[block]][1];

                if(P[2][1] > (rel_ymax4<<shift)) P[2][1]= (rel_ymax4<<shift);

                if(P[3][0] < (rel_xmin4<<shift)) P[3][0]= (rel_xmin4<<shift);

                if(P[3][0] > (rel_xmax4<<shift)) P[3][0]= (rel_xmax4<<shift);

                if(P[3][1] > (rel_ymax4<<shift)) P[3][1]= (rel_ymax4<<shift);

        

                P[4][0]= mid_pred(P[1][0], P[2][0], P[3][0]);

                P[4][1]= mid_pred(P[1][1], P[2][1], P[3][1]);

            }

            if(s->out_format == FMT_H263){

                pred_x4 = P[4][0];

                pred_y4 = P[4][1];

            }else { /* mpeg1 at least */

                pred_x4= P[1][0];

                pred_y4= P[1][1];

            }

            P[5][0]= mx - mb_x*16;

            P[5][1]= my - mb_y*16;



            dmin4 = epzs_motion_search4(s, block, &mx4, &my4, P, pred_x4, pred_y4, rel_xmin4, rel_ymin4, rel_xmax4, rel_ymax4, ref_picture);



            halfpel_motion_search4(s, &mx4, &my4, dmin4, rel_xmin4, rel_ymin4, rel_xmax4, rel_ymax4, 

                                   pred_x4, pred_y4, block_x, block_y, ref_picture);

     

            s->motion_val[ s->block_index[block] ][0]= mx4;

            s->motion_val[ s->block_index[block] ][1]= my4;

        }

    }



    /* intra / predictive decision */

    xx = mb_x * 16;

    yy = mb_y * 16;



    pix = s->new_picture[0] + (yy * s->linesize) + xx;

    /* At this point (mx,my) are full-pell and the absolute displacement */

    ppix = ref_picture + (my * s->linesize) + mx;

    

    sum = pix_sum(pix, s->linesize);

#if 0

    varc = pix_dev(pix, s->linesize, (sum+128)>>8) + INTER_BIAS;

    vard = pix_abs16x16(pix, ppix, s->linesize);

#else

    sum= (sum+8)>>4;

    varc = ((pix_norm1(pix, s->linesize) - sum*sum + 128 + 500)>>8);

    vard = (pix_norm(pix, ppix, s->linesize)+128)>>8;

#endif



    s->mb_var[s->mb_width * mb_y + mb_x] = varc;

    s->avg_mb_var+= varc;

    s->mc_mb_var += vard;



    

#if 0

    printf("varc=%4d avg_var=%4d (sum=%4d) vard=%4d mx=%2d my=%2d\n",

	   varc, s->avg_mb_var, sum, vard, mx - xx, my - yy);

#endif

    if(s->flags&CODEC_FLAG_HQ){

        if (vard*2 + 200 > varc)

            mb_type|= MB_TYPE_INTRA;

        if (varc*2 + 200 > vard){

            mb_type|= MB_TYPE_INTER;

            halfpel_motion_search(s, &mx, &my, dmin, xmin, ymin, xmax, ymax, pred_x, pred_y, ref_picture);

        }else{

            mx = mx*2 - mb_x*32;

            my = my*2 - mb_y*32;

        }

    }else{

        if (vard <= 64 || vard < varc) {

            mb_type|= MB_TYPE_INTER;

            if (s->me_method != ME_ZERO) {

                halfpel_motion_search(s, &mx, &my, dmin, xmin, ymin, xmax, ymax, pred_x, pred_y, ref_picture);

            } else {

                mx -= 16 * mb_x;

                my -= 16 * mb_y;

            }

#if 0

            if (vard < 10) {

                skip++;

                fprintf(stderr,"\nEarly skip: %d vard: %2d varc: %5d dmin: %d", 

                                skip, vard, varc, dmin);

            }

#endif

        }else{

            mb_type|= MB_TYPE_INTRA;

            mx = 0;//mx*2 - 32 * mb_x;

            my = 0;//my*2 - 32 * mb_y;

        }

    }



    s->mb_type[mb_y*s->mb_width + mb_x]= mb_type;

    set_p_mv_tables(s, mx, my);

}
