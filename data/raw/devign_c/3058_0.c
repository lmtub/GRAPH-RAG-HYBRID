void qemu_spice_destroy_primary_surface(SimpleSpiceDisplay *ssd,

                                        uint32_t id, qxl_async_io async)

{

    if (async != QXL_SYNC) {

#if SPICE_INTERFACE_QXL_MINOR >= 1

        spice_qxl_destroy_primary_surface_async(&ssd->qxl, id, 0);

#else

        abort();

#endif

    } else {

        ssd->worker->destroy_primary_surface(ssd->worker, id);

    }

}
