GSource *iohandler_get_g_source(void)

{

    iohandler_init();

    return aio_get_g_source(iohandler_ctx);

}
