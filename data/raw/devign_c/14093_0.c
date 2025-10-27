static void pollfds_cleanup(Notifier *n, void *unused)

{

    g_assert(npfd == 0);

    g_free(pollfds);

    g_free(nodes);

    nalloc = 0;

}
