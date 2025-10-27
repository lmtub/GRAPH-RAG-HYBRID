static void apply_motion_4x4(RoqContext *ri, int x, int y, unsigned char mv,

    signed char mean_x, signed char mean_y)

{

    int i, hw, mx, my;

    unsigned char *pa, *pb;



    mx = x + 8 - (mv >> 4) - mean_x;

    my = y + 8 - (mv & 0xf) - mean_y;



    pa = ri->current_frame.data[0] + (y * ri->y_stride) + x;

    pb = ri->last_frame.data[0] + (my * ri->y_stride) + mx;

    for(i = 0; i < 4; i++) {

        pa[0] = pb[0];

        pa[1] = pb[1];

        pa[2] = pb[2];

        pa[3] = pb[3];

        pa += ri->y_stride;

        pb += ri->y_stride;

    }



#if 0

    pa = ri->current_frame.data[1] + (y/2) * (ri->c_stride) + x/2;

    pb = ri->last_frame.data[1] + (my/2) * (ri->c_stride) + (mx + 1)/2;

    for(i = 0; i < 2; i++) {

        pa[0] = pb[0];

        pa[1] = pb[1];

        pa += ri->c_stride;

        pb += ri->c_stride;

    }



    pa = ri->current_frame.data[2] + (y/2) * (ri->c_stride) + x/2;

    pb = ri->last_frame.data[2] + (my/2) * (ri->c_stride) + (mx + 1)/2;

    for(i = 0; i < 2; i++) {

        pa[0] = pb[0];

        pa[1] = pb[1];

        pa += ri->c_stride;

        pb += ri->c_stride;

    }

#else

    hw = ri->y_stride/2;

    pa = ri->current_frame.data[1] + (y * ri->y_stride)/4 + x/2;

    pb = ri->last_frame.data[1] + (my/2) * (ri->y_stride/2) + (mx + 1)/2;



    for(i = 0; i < 2; i++) {

        switch(((my & 0x01) << 1) | (mx & 0x01)) {



        case 0:

            pa[0] = pb[0];

            pa[1] = pb[1];

            pa[hw] = pb[hw];

            pa[hw+1] = pb[hw+1];

            break;



        case 1:

            pa[0] = avg2(pb[0], pb[1]);

            pa[1] = avg2(pb[1], pb[2]);

            pa[hw] = avg2(pb[hw], pb[hw+1]);

            pa[hw+1] = avg2(pb[hw+1], pb[hw+2]);

            break;



        case 2:

            pa[0] = avg2(pb[0], pb[hw]);

            pa[1] = avg2(pb[1], pb[hw+1]);

            pa[hw] = avg2(pb[hw], pb[hw*2]);

            pa[hw+1] = avg2(pb[hw+1], pb[(hw*2)+1]);

            break;



        case 3:

            pa[0] = avg4(pb[0], pb[1], pb[hw], pb[hw+1]);

            pa[1] = avg4(pb[1], pb[2], pb[hw+1], pb[hw+2]);

            pa[hw] = avg4(pb[hw], pb[hw+1], pb[hw*2], pb[(hw*2)+1]);

            pa[hw+1] = avg4(pb[hw+1], pb[hw+2], pb[(hw*2)+1], pb[(hw*2)+1]);

            break;

        }



        pa = ri->current_frame.data[2] + (y * ri->y_stride)/4 + x/2;

        pb = ri->last_frame.data[2] + (my/2) * (ri->y_stride/2) + (mx + 1)/2;

    }

#endif

}
