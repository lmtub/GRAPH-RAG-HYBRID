iscsi_process_read(void *arg)

{

    IscsiLun *iscsilun = arg;

    struct iscsi_context *iscsi = iscsilun->iscsi;



    aio_context_acquire(iscsilun->aio_context);

    iscsi_service(iscsi, POLLIN);

    iscsi_set_events(iscsilun);

    aio_context_release(iscsilun->aio_context);

}
