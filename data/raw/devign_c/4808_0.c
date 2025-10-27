static int init_dimensions(H264Context *h)

{

    int width  = h->width  - (h->sps.crop_right + h->sps.crop_left);

    int height = h->height - (h->sps.crop_top   + h->sps.crop_bottom);



    /* handle container cropping */

    if (FFALIGN(h->avctx->width,  16) == FFALIGN(width,  16) &&

        FFALIGN(h->avctx->height, 16) == FFALIGN(height, 16)) {

        width  = h->avctx->width;

        height = h->avctx->height;

    }



    if (width <= 0 || height <= 0) {

        av_log(h->avctx, AV_LOG_ERROR, "Invalid cropped dimensions: %dx%d.\n",

               width, height);

        if (h->avctx->err_recognition & AV_EF_EXPLODE)

            return AVERROR_INVALIDDATA;



        av_log(h->avctx, AV_LOG_WARNING, "Ignoring cropping information.\n");

        h->sps.crop_bottom =

        h->sps.crop_top    =

        h->sps.crop_right  =

        h->sps.crop_left   =

        h->sps.crop        = 0;



        width  = h->width;

        height = h->height;

    }



    h->avctx->coded_width  = h->width;

    h->avctx->coded_height = h->height;

    h->avctx->width        = width;

    h->avctx->height       = height;



    return 0;

}
