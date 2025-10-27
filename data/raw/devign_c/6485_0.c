static void start_frame(AVFilterLink *link, AVFilterPicRef *picref)

{

    CropContext *crop = link->dst->priv;

    AVFilterPicRef *ref2 = avfilter_ref_pic(picref, ~0);

    int i;



    ref2->w        = crop->w;

    ref2->h        = crop->h;



    ref2->data[0] += crop->y * ref2->linesize[0];

    ref2->data[0] += (crop->x * crop->bpp) >> 3;



    if (link->format != PIX_FMT_PAL8      &&

        link->format != PIX_FMT_BGR4_BYTE &&

        link->format != PIX_FMT_RGB4_BYTE &&

        link->format != PIX_FMT_BGR8      &&

        link->format != PIX_FMT_RGB8) {

        for (i = 1; i < 3; i ++) {

            if (ref2->data[i]) {

                ref2->data[i] += (crop->y >> crop->vsub) * ref2->linesize[i];

                ref2->data[i] += ((crop->x * crop->bpp) >> 3) >> crop->hsub;

            }

        }

    }



    /* alpha plane */

    if (ref2->data[3]) {

        ref2->data[3] += crop->y * ref2->linesize[3];

        ref2->data[3] += (crop->x * crop->bpp) >> 3;

    }



    avfilter_start_frame(link->dst->outputs[0], ref2);

}
