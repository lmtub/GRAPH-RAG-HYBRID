void rdma_start_outgoing_migration(void *opaque,

                            const char *host_port, Error **errp)

{

    MigrationState *s = opaque;

    Error *local_err = NULL, **temp = &local_err;

    RDMAContext *rdma = qemu_rdma_data_init(host_port, &local_err);

    int ret = 0;



    if (rdma == NULL) {

        ERROR(temp, "Failed to initialize RDMA data structures! %d", ret);

        goto err;

    }



    ret = qemu_rdma_source_init(rdma, &local_err,

        s->enabled_capabilities[MIGRATION_CAPABILITY_RDMA_PIN_ALL]);



    if (ret) {

        goto err;

    }



    trace_rdma_start_outgoing_migration_after_rdma_source_init();

    ret = qemu_rdma_connect(rdma, &local_err);



    if (ret) {

        goto err;

    }



    trace_rdma_start_outgoing_migration_after_rdma_connect();



    s->to_dst_file = qemu_fopen_rdma(rdma, "wb");

    migrate_fd_connect(s);

    return;

err:

    error_propagate(errp, local_err);

    g_free(rdma);

    migrate_fd_error(s);

}
