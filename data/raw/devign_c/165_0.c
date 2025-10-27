void ga_channel_free(GAChannel *c)

{

    if (c->method == GA_CHANNEL_UNIX_LISTEN

        && c->listen_channel) {

        ga_channel_listen_close(c);

    }

    if (c->client_channel) {

        ga_channel_client_close(c);

    }

    g_free(c);

}
