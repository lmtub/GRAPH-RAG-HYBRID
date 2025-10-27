static void io_watch_poll_finalize(GSource *source)

{

    IOWatchPoll *iwp = io_watch_poll_from_source(source);

    g_source_destroy(iwp->src);

    g_source_unref(iwp->src);

    iwp->src = NULL;

}
