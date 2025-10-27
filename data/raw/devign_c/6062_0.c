int ff_h264_execute_decode_slices(H264Context *h, unsigned context_count)

{

    AVCodecContext *const avctx = h->avctx;

    H264Context *hx;

    int i;



    if (h->mb_y >= h->mb_height) {

        av_log(h->avctx, AV_LOG_ERROR,

               "Input contains more MB rows than the frame height.\n");

        return AVERROR_INVALIDDATA;

    }



    if (h->avctx->hwaccel)

        return 0;

    if (context_count == 1) {

        return decode_slice(avctx, &h);

    } else {

        for (i = 1; i < context_count; i++) {

            hx                 = h->thread_context[i];

            hx->er.error_count = 0;

        }



        avctx->execute(avctx, decode_slice, h->thread_context,

                       NULL, context_count, sizeof(void *));



        /* pull back stuff from slices to master context */

        hx                   = h->thread_context[context_count - 1];

        h->mb_x              = hx->mb_x;

        h->mb_y              = hx->mb_y;

        h->droppable         = hx->droppable;

        h->picture_structure = hx->picture_structure;

        for (i = 1; i < context_count; i++)

            h->er.error_count += h->thread_context[i]->er.error_count;

    }



    return 0;

}
