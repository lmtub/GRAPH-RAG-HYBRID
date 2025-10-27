static int get_buffer_sao(HEVCContext *s, AVFrame *frame, const HEVCSPS *sps)

{

    int ret, i;



    frame->width  = s->avctx->width  + 2;

    frame->height = s->avctx->height + 2;

    if ((ret = ff_get_buffer(s->avctx, frame, AV_GET_BUFFER_FLAG_REF)) < 0)

        return ret;

    for (i = 0; frame->data[i]; i++) {

        int offset = frame->linesize[i] + (1 << sps->pixel_shift);

        frame->data[i] += offset;

    }

    frame->width  = s->avctx->width;

    frame->height = s->avctx->height;



    return 0;

}
