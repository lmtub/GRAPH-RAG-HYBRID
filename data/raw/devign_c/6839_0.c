AioContext *aio_context_new(void)

{

    return (AioContext *) g_source_new(&aio_source_funcs, sizeof(AioContext));

}
