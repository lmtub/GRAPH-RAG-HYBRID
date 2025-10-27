static void ga_channel_listen_close(GAChannel *c)

{

    g_assert(c->method == GA_CHANNEL_UNIX_LISTEN);

    g_assert(c->listen_channel);

    g_io_channel_shutdown(c->listen_channel, true, NULL);

    g_io_channel_unref(c->listen_channel);

    c->listen_channel = NULL;

}
