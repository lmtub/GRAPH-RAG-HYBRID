void scsi_req_unref(SCSIRequest *req)

{

    assert(req->refcount > 0);

    if (--req->refcount == 0) {

        BusState *qbus = req->dev->qdev.parent_bus;

        SCSIBus *bus = DO_UPCAST(SCSIBus, qbus, qbus);



        if (bus->info->free_request && req->hba_private) {

            bus->info->free_request(bus, req->hba_private);

        }

        if (req->ops->free_req) {

            req->ops->free_req(req);

        }

        object_unref(OBJECT(req->dev));

        object_unref(OBJECT(qbus->parent));

        g_free(req);

    }

}
