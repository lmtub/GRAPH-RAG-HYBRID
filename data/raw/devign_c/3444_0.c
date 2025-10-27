aio_ctx_dispatch(GSource     *source,

                 GSourceFunc  callback,

                 gpointer     user_data)

{

    AioContext *ctx = (AioContext *) source;



    assert(callback == NULL);

    aio_dispatch(ctx, true);

    return true;

}
