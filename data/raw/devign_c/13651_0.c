static inline void apply_motion_generic(RoqContext *ri, int x, int y, int deltax,

                                        int deltay, int sz)

{

    int mx, my, cp;



    mx = x + deltax;

    my = y + deltay;



    /* check MV against frame boundaries */

    if ((mx < 0) || (mx > ri->width - sz) ||

        (my < 0) || (my > ri->height - sz)) {

        av_log(ri->avctx, AV_LOG_ERROR, "motion vector out of bounds: MV = (%d, %d), boundaries = (0, 0, %d, %d)\n",

            mx, my, ri->width, ri->height);

        return;

    }



    if (ri->last_frame->data[0] == NULL) {

        av_log(ri->avctx, AV_LOG_ERROR, "Invalid decode type. Invalid header?\n");

        return;

    }



    for(cp = 0; cp < 3; cp++) {

        int outstride = ri->current_frame->linesize[cp];

        int instride  = ri->last_frame   ->linesize[cp];

        block_copy(ri->current_frame->data[cp] + y*outstride + x,

                   ri->last_frame->data[cp] + my*instride + mx,

                   outstride, instride, sz);

    }

}
