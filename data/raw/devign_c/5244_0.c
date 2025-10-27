static gboolean io_watch_poll_prepare(GSource *source, gint *timeout_)

{

    IOWatchPoll *iwp = io_watch_poll_from_source(source);

    bool now_active = iwp->fd_can_read(iwp->opaque) > 0;

    bool was_active = g_source_get_context(iwp->src) != NULL;

    if (was_active == now_active) {

        return FALSE;

    }



    if (now_active) {

        g_source_attach(iwp->src, NULL);

    } else {

        g_source_remove(g_source_get_id(iwp->src));

    }

    return FALSE;

}
