static gboolean ga_channel_listen_accept(GIOChannel *channel,

                                         GIOCondition condition, gpointer data)

{

    GAChannel *c = data;

    int ret, client_fd;

    bool accepted = false;

    struct sockaddr_un addr;

    socklen_t addrlen = sizeof(addr);



    g_assert(channel != NULL);



    client_fd = qemu_accept(g_io_channel_unix_get_fd(channel),

                            (struct sockaddr *)&addr, &addrlen);

    if (client_fd == -1) {

        g_warning("error converting fd to gsocket: %s", strerror(errno));

        goto out;

    }

    fcntl(client_fd, F_SETFL, O_NONBLOCK);

    ret = ga_channel_client_add(c, client_fd);

    if (ret) {

        g_warning("error setting up connection");


        goto out;

    }

    accepted = true;



out:

    /* only accept 1 connection at a time */

    return !accepted;

}