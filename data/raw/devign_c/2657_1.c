static void qxl_hard_reset(PCIQXLDevice *d, int loadvm)

{

    dprint(d, 1, "%s: start%s\n", __FUNCTION__,

           loadvm ? " (loadvm)" : "");



    qemu_mutex_unlock_iothread();

    d->ssd.worker->reset_cursor(d->ssd.worker);

    d->ssd.worker->reset_image_cache(d->ssd.worker);

    qemu_mutex_lock_iothread();

    qxl_reset_surfaces(d);

    qxl_reset_memslots(d);



    /* pre loadvm reset must not touch QXLRam.  This lives in

     * device memory, is migrated together with RAM and thus

     * already loaded at this point */

    if (!loadvm) {

        qxl_reset_state(d);

    }

    qemu_spice_create_host_memslot(&d->ssd);

    qxl_soft_reset(d);



    dprint(d, 1, "%s: done\n", __FUNCTION__);

}
