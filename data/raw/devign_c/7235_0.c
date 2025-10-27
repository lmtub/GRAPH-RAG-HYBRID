static void apply_motion_8x8(RoqContext *ri, int x, int y,

    unsigned char mv, signed char mean_x, signed char mean_y)

{

    int mx, my, i, j, hw;

    unsigned char *pa, *pb;



    mx = x + 8 - (mv >> 4) - mean_x;

    my = y + 8 - (mv & 0xf) - mean_y;



    pa = ri->current_frame.data[0] + (y * ri->y_stride) + x;

    pb = ri->last_frame.data[0] + (my * ri->y_stride) + mx;

    for(i = 0; i < 8; i++) {

        pa[0] = pb[0];

        pa[1] = pb[1];

        pa[2] = pb[2];

        pa[3] = pb[3];

        pa[4] = pb[4];

        pa[5] = pb[5];

        pa[6] = pb[6];

        pa[7] = pb[7];

        pa += ri->y_stride;

        pb += ri->y_stride;

    }



#if 0

    pa = ri->current_frame.data[1] + (y/2) * (ri->c_stride) + x/2;

    pb = ri->last_frame.data[1] + (my/2) * (ri->c_stride) + (mx + 1)/2;

    for(i = 0; i < 4; i++) {

        pa[0] = pb[0];

        pa[1] = pb[1];

        pa[2] = pb[2];

        pa[3] = pb[3];

        pa += ri->c_stride;

        pb += ri->c_stride;

    }



    pa = ri->current_frame.data[2] + (y/2) * (ri->c_stride) + x/2;

    pb = ri->last_frame.data[2] + (my/2) * (ri->c_stride) + (mx + 1)/2;

    for(i = 0; i < 4; i++) {

        pa[0] = pb[0];

        pa[1] = pb[1];

        pa[2] = pb[2];

        pa[3] = pb[3];

        pa += ri->c_stride;

        pb += ri->c_stride;

    }

#else

    hw = ri->c_stride;

    pa = ri->current_frame.data[1] + (y * ri->y_stride)/4 + x/2;

    pb = ri->last_frame.data[1] + (my/2) * (ri->y_stride/2) + (mx + 1)/2;

    for(j = 0; j < 2; j++) {

        for(i = 0; i < 4; i++) {

            switch(((my & 0x01) << 1) | (mx & 0x01)) {



            case 0:

                pa[0] = pb[0];

                pa[1] = pb[1];

                pa[2] = pb[2];

                pa[3] = pb[3];

                break;



            case 1:

                pa[0] = avg2(pb[0], pb[1]);

                pa[1] = avg2(pb[1], pb[2]);

                pa[2] = avg2(pb[2], pb[3]);

                pa[3] = avg2(pb[3], pb[4]);

                break;

 

            case 2:

                pa[0] = avg2(pb[0], pb[hw]);

                pa[1] = avg2(pb[1], pb[hw+1]);

                pa[2] = avg2(pb[2], pb[hw+2]);

                pa[3] = avg2(pb[3], pb[hw+3]);

                break;



            case 3:

                pa[0] = avg4(pb[0], pb[1], pb[hw], pb[hw+1]);

                pa[1] = avg4(pb[1], pb[2], pb[hw+1], pb[hw+2]);

                pa[2] = avg4(pb[2], pb[3], pb[hw+2], pb[hw+3]);

                pa[3] = avg4(pb[3], pb[4], pb[hw+3], pb[hw+4]);

                break;

            }

            pa += ri->c_stride;

            pb += ri->c_stride;

        }



        pa = ri->current_frame.data[2] + (y * ri->y_stride)/4 + x/2;

        pb = ri->last_frame.data[2] + (my/2) * (ri->y_stride/2) + (mx + 1)/2;

    }

#endif

}
