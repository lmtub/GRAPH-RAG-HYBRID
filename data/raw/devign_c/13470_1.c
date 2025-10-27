void qemu_spice_destroy_host_primary(SimpleSpiceDisplay *ssd)

{

    dprint(1, "%s:\n", __FUNCTION__);



    qemu_mutex_unlock_iothread();

    ssd->worker->destroy_primary_surface(ssd->worker, 0);

    qemu_mutex_lock_iothread();

}
