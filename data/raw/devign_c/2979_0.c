void bdrv_info_stats(Monitor *mon, QObject **ret_data)

{

    QObject *obj;

    QList *devices;

    BlockDriverState *bs;



    devices = qlist_new();



    for (bs = bdrv_first; bs != NULL; bs = bs->next) {

        obj = qobject_from_jsonf("{ 'device': %s, 'stats': {"

                                 "'rd_bytes': %" PRId64 ","

                                 "'wr_bytes': %" PRId64 ","

                                 "'rd_operations': %" PRId64 ","

                                 "'wr_operations': %" PRId64

                                 "} }",

                                 bs->device_name,

                                 bs->rd_bytes, bs->wr_bytes,

                                 bs->rd_ops, bs->wr_ops);

        assert(obj != NULL);

        qlist_append_obj(devices, obj);

    }



    *ret_data = QOBJECT(devices);

}
