void qio_dns_resolver_lookup_async(QIODNSResolver *resolver,

                                   SocketAddress *addr,

                                   QIOTaskFunc func,

                                   gpointer opaque,

                                   GDestroyNotify notify)

{

    QIOTask *task;

    struct QIODNSResolverLookupData *data =

        g_new0(struct QIODNSResolverLookupData, 1);



    data->addr = QAPI_CLONE(SocketAddress, addr);



    task = qio_task_new(OBJECT(resolver), func, opaque, notify);



    qio_task_run_in_thread(task,

                           qio_dns_resolver_lookup_worker,

                           data,

                           qio_dns_resolver_lookup_data_free);

}
