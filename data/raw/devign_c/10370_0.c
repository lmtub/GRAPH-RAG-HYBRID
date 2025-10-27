static void vnc_client_cache_addr(VncState *client)

{

    Error *err = NULL;



    client->info = g_malloc0(sizeof(*client->info));

    client->info->base = g_malloc0(sizeof(*client->info->base));

    vnc_init_basic_info_from_remote_addr(client->csock, client->info->base,

                                         &err);

    if (err) {

        qapi_free_VncClientInfo(client->info);

        client->info = NULL;

        error_free(err);

    }

}
