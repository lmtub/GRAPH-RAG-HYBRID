static int net_socket_connect_init(NetClientState *peer,

                                   const char *model,

                                   const char *name,

                                   const char *host_str)

{

    socket_connect_data *c = g_new0(socket_connect_data, 1);

    int fd = -1;

    Error *local_error = NULL;



    c->peer = peer;

    c->model = g_strdup(model);

    c->name = g_strdup(name);

    c->saddr = socket_parse(host_str, &local_error);

    if (c->saddr == NULL) {

        goto err;

    }



    fd = socket_connect(c->saddr, net_socket_connected, c, &local_error);

    if (fd < 0) {

        goto err;

    }



    return 0;



err:

    error_report_err(local_error);

    socket_connect_data_free(c);

    return -1;

}
