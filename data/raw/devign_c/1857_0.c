void cpu_unregister_map_client(void *_client)

{

    MapClient *client = (MapClient *)_client;



    LIST_REMOVE(client, link);

    qemu_free(client);

}
