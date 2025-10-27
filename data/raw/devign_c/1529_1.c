static void cpu_notify_map_clients_locked(void)

{

    MapClient *client;



    while (!QLIST_EMPTY(&map_client_list)) {

        client = QLIST_FIRST(&map_client_list);

        client->callback(client->opaque);

        cpu_unregister_map_client(client);

    }

}
