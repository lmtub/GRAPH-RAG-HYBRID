static AVBufferRef *dxva2_pool_alloc(void *opaque, int size)

{

    AVHWFramesContext      *ctx = (AVHWFramesContext*)opaque;

    DXVA2FramesContext       *s = ctx->internal->priv;

    AVDXVA2FramesContext *hwctx = ctx->hwctx;



    if (s->nb_surfaces_used < hwctx->nb_surfaces) {

        s->nb_surfaces_used++;

        return av_buffer_create((uint8_t*)s->surfaces_internal[s->nb_surfaces_used - 1],

                                sizeof(*hwctx->surfaces), NULL, 0, 0);

    }



    return NULL;

}
