void inter_predict(VP8Context *s, uint8_t *dst[3], VP8Macroblock *mb,

                   int mb_x, int mb_y)

{

    int x_off = mb_x << 4, y_off = mb_y << 4;

    int width = 16*s->mb_width, height = 16*s->mb_height;

    AVFrame *ref = s->framep[mb->ref_frame];

    VP56mv *bmv = mb->bmv;



    switch (mb->partitioning) {

    case VP8_SPLITMVMODE_NONE:

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    0, 0, 16, 16, width, height, &mb->mv);

        break;

    case VP8_SPLITMVMODE_4x4: {

        int x, y;

        VP56mv uvmv;



        /* Y */

        for (y = 0; y < 4; y++) {

            for (x = 0; x < 4; x++) {

                vp8_mc(s, 1, dst[0] + 4*y*s->linesize + x*4,

                       ref->data[0], &bmv[4*y + x],

                       4*x + x_off, 4*y + y_off, 4, 4,

                       width, height, s->linesize,

                       s->put_pixels_tab[2]);

            }

        }



        /* U/V */

        x_off >>= 1; y_off >>= 1; width >>= 1; height >>= 1;

        for (y = 0; y < 2; y++) {

            for (x = 0; x < 2; x++) {

                uvmv.x = mb->bmv[ 2*y    * 4 + 2*x  ].x +

                         mb->bmv[ 2*y    * 4 + 2*x+1].x +

                         mb->bmv[(2*y+1) * 4 + 2*x  ].x +

                         mb->bmv[(2*y+1) * 4 + 2*x+1].x;

                uvmv.y = mb->bmv[ 2*y    * 4 + 2*x  ].y +

                         mb->bmv[ 2*y    * 4 + 2*x+1].y +

                         mb->bmv[(2*y+1) * 4 + 2*x  ].y +

                         mb->bmv[(2*y+1) * 4 + 2*x+1].y;

                uvmv.x = (uvmv.x + 2 + (uvmv.x >> (INT_BIT-1))) >> 2;

                uvmv.y = (uvmv.y + 2 + (uvmv.y >> (INT_BIT-1))) >> 2;

                if (s->profile == 3) {

                    uvmv.x &= ~7;

                    uvmv.y &= ~7;

                }

                vp8_mc(s, 0, dst[1] + 4*y*s->uvlinesize + x*4,

                       ref->data[1], &uvmv,

                       4*x + x_off, 4*y + y_off, 4, 4,

                       width, height, s->uvlinesize,

                       s->put_pixels_tab[2]);

                vp8_mc(s, 0, dst[2] + 4*y*s->uvlinesize + x*4,

                       ref->data[2], &uvmv,

                       4*x + x_off, 4*y + y_off, 4, 4,

                       width, height, s->uvlinesize,

                       s->put_pixels_tab[2]);

            }

        }

        break;

    }

    case VP8_SPLITMVMODE_16x8:

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    0, 0, 16, 8, width, height, &bmv[0]);

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    0, 8, 16, 8, width, height, &bmv[1]);

        break;

    case VP8_SPLITMVMODE_8x16:

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    0, 0, 8, 16, width, height, &bmv[0]);

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    8, 0, 8, 16, width, height, &bmv[1]);

        break;

    case VP8_SPLITMVMODE_8x8:

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    0, 0, 8, 8, width, height, &bmv[0]);

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    8, 0, 8, 8, width, height, &bmv[1]);

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    0, 8, 8, 8, width, height, &bmv[2]);

        vp8_mc_part(s, dst, ref, x_off, y_off,

                    8, 8, 8, 8, width, height, &bmv[3]);

        break;

    }

}
