static void qxl_spice_destroy_surface_wait(PCIQXLDevice *qxl, uint32_t id,

                                           qxl_async_io async)

{

    if (async) {

#if SPICE_INTERFACE_QXL_MINOR < 1

        abort();

#else

        spice_qxl_destroy_surface_async(&qxl->ssd.qxl, id,

                                        (uint64_t)id);

#endif

    } else {

        qxl->ssd.worker->destroy_surface_wait(qxl->ssd.worker, id);

        qxl_spice_destroy_surface_wait_complete(qxl, id);

    }

}
