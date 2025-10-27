iscsi_set_events(IscsiLun *iscsilun)

{

    struct iscsi_context *iscsi = iscsilun->iscsi;

    int ev;



    /* We always register a read handler.  */

    ev = POLLIN;

    ev |= iscsi_which_events(iscsi);

    if (ev != iscsilun->events) {

        aio_set_fd_handler(iscsilun->aio_context,

                           iscsi_get_fd(iscsi),

                           iscsi_process_read,

                           (ev & POLLOUT) ? iscsi_process_write : NULL,

                           iscsilun);



    }



    iscsilun->events = ev;

}
