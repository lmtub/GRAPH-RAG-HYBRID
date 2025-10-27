void qemu_spice_add_memslot(SimpleSpiceDisplay *ssd, QXLDevMemSlot *memslot,

                            qxl_async_io async)

{

    if (async != QXL_SYNC) {

#if SPICE_INTERFACE_QXL_MINOR >= 1

        spice_qxl_add_memslot_async(&ssd->qxl, memslot, 0);

#else

        abort();

#endif

    } else {

        ssd->worker->add_memslot(ssd->worker, memslot);

    }

}
