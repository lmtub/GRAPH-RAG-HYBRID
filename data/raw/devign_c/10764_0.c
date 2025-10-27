static int RENAME(epzs_motion_search2)(MpegEncContext * s,

                             int *mx_ptr, int *my_ptr,

                             int P[10][2], int pred_x, int pred_y,

                             uint8_t *src_data[3], 

                             uint8_t *ref_data[3], int stride, int uvstride, int16_t (*last_mv)[2], 

                             int ref_mv_scale, uint8_t * const mv_penalty)

{

    int best[2]={0, 0};

    int d, dmin; 

    const int shift= 1+s->quarter_sample;

    uint32_t *map= s->me.map;

    int map_generation;

    const int penalty_factor= s->me.penalty_factor;

    const int size=0; //FIXME pass as arg

    const int h=8;

    const int ref_mv_stride= s->mb_stride;

    const int ref_mv_xy= s->mb_x + s->mb_y *ref_mv_stride;

    me_cmp_func cmp, chroma_cmp;

    LOAD_COMMON

    

    cmp= s->dsp.me_cmp[size];

    chroma_cmp= s->dsp.me_cmp[size+1];



    map_generation= update_map_generation(s);



    dmin = 1000000;

//printf("%d %d %d %d //",xmin, ymin, xmax, ymax); 

    /* first line */

    if (s->first_slice_line) {

	CHECK_MV(P_LEFT[0]>>shift, P_LEFT[1]>>shift)

        CHECK_CLIPED_MV((last_mv[ref_mv_xy][0]*ref_mv_scale + (1<<15))>>16, 

                        (last_mv[ref_mv_xy][1]*ref_mv_scale + (1<<15))>>16)

        CHECK_MV(P_MV1[0]>>shift, P_MV1[1]>>shift)

    }else{

        CHECK_MV(P_MV1[0]>>shift, P_MV1[1]>>shift)

        //FIXME try some early stop

        if(dmin>64*2){

            CHECK_MV(P_MEDIAN[0]>>shift, P_MEDIAN[1]>>shift)

            CHECK_MV(P_LEFT[0]>>shift, P_LEFT[1]>>shift)

            CHECK_MV(P_TOP[0]>>shift, P_TOP[1]>>shift)

            CHECK_MV(P_TOPRIGHT[0]>>shift, P_TOPRIGHT[1]>>shift)

            CHECK_CLIPED_MV((last_mv[ref_mv_xy][0]*ref_mv_scale + (1<<15))>>16, 

                            (last_mv[ref_mv_xy][1]*ref_mv_scale + (1<<15))>>16)

        }

    }

    if(dmin>64*4){

        CHECK_CLIPED_MV((last_mv[ref_mv_xy+1][0]*ref_mv_scale + (1<<15))>>16, 

                        (last_mv[ref_mv_xy+1][1]*ref_mv_scale + (1<<15))>>16)

        if(s->end_mb_y == s->mb_height || s->mb_y+1<s->end_mb_y)  //FIXME replace at least with last_slice_line

            CHECK_CLIPED_MV((last_mv[ref_mv_xy+ref_mv_stride][0]*ref_mv_scale + (1<<15))>>16, 

                            (last_mv[ref_mv_xy+ref_mv_stride][1]*ref_mv_scale + (1<<15))>>16)

    }



    if(s->me.dia_size==-1)

        dmin= RENAME(funny_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);

    else if(s->me.dia_size<-1)

        dmin= RENAME(sab_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);

    else if(s->me.dia_size<2)

        dmin= RENAME(small_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);

    else

        dmin= RENAME(var_diamond_search)(s, best, dmin, src_data, ref_data, stride, uvstride,

                                   pred_x, pred_y, penalty_factor, 

				   shift, map, map_generation, size, h, mv_penalty);





    *mx_ptr= best[0];

    *my_ptr= best[1];    



//    printf("%d %d %d \n", best[0], best[1], dmin);

    return dmin;

}
