static int sad_hpel_motion_search(MpegEncContext * s,

                                  int *mx_ptr, int *my_ptr, int dmin,

                                  int src_index, int ref_index,

                                  int size, int h)

{

    MotionEstContext * const c= &s->me;

    const int penalty_factor= c->sub_penalty_factor;

    int mx, my, dminh;

    uint8_t *pix, *ptr;

    int stride= c->stride;

    LOAD_COMMON



    av_assert2(c->sub_flags == 0);



    if(c->skip){

        *mx_ptr = 0;

        *my_ptr = 0;

        return dmin;

    }



    pix = c->src[src_index][0];



    mx = *mx_ptr;

    my = *my_ptr;

    ptr = c->ref[ref_index][0] + (my * stride) + mx;



    dminh = dmin;



    if (mx > xmin && mx < xmax &&

        my > ymin && my < ymax) {

        int dx=0, dy=0;

        int d, pen_x, pen_y;

        const int index= (my<<ME_MAP_SHIFT) + mx;

        const int t= score_map[(index-(1<<ME_MAP_SHIFT))&(ME_MAP_SIZE-1)];

        const int l= score_map[(index- 1               )&(ME_MAP_SIZE-1)];

        const int r= score_map[(index+ 1               )&(ME_MAP_SIZE-1)];

        const int b= score_map[(index+(1<<ME_MAP_SHIFT))&(ME_MAP_SIZE-1)];

        mx<<=1;

        my<<=1;





        pen_x= pred_x + mx;

        pen_y= pred_y + my;



        ptr-= stride;

        if(t<=b){

            CHECK_SAD_HALF_MV(y2 , 0, -1)

            if(l<=r){

                CHECK_SAD_HALF_MV(xy2, -1, -1)

                if(t+r<=b+l){

                    CHECK_SAD_HALF_MV(xy2, +1, -1)

                    ptr+= stride;

                }else{

                    ptr+= stride;

                    CHECK_SAD_HALF_MV(xy2, -1, +1)

                }

                CHECK_SAD_HALF_MV(x2 , -1,  0)

            }else{

                CHECK_SAD_HALF_MV(xy2, +1, -1)

                if(t+l<=b+r){

                    CHECK_SAD_HALF_MV(xy2, -1, -1)

                    ptr+= stride;

                }else{

                    ptr+= stride;

                    CHECK_SAD_HALF_MV(xy2, +1, +1)

                }

                CHECK_SAD_HALF_MV(x2 , +1,  0)

            }

        }else{

            if(l<=r){

                if(t+l<=b+r){

                    CHECK_SAD_HALF_MV(xy2, -1, -1)

                    ptr+= stride;

                }else{

                    ptr+= stride;

                    CHECK_SAD_HALF_MV(xy2, +1, +1)

                }

                CHECK_SAD_HALF_MV(x2 , -1,  0)

                CHECK_SAD_HALF_MV(xy2, -1, +1)

            }else{

                if(t+r<=b+l){

                    CHECK_SAD_HALF_MV(xy2, +1, -1)

                    ptr+= stride;

                }else{

                    ptr+= stride;

                    CHECK_SAD_HALF_MV(xy2, -1, +1)

                }

                CHECK_SAD_HALF_MV(x2 , +1,  0)

                CHECK_SAD_HALF_MV(xy2, +1, +1)

            }

            CHECK_SAD_HALF_MV(y2 ,  0, +1)

        }

        mx+=dx;

        my+=dy;



    }else{

        mx<<=1;

        my<<=1;

    }



    *mx_ptr = mx;

    *my_ptr = my;

    return dminh;

}
