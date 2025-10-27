void qemu_spice_vm_change_state_handler(void *opaque, int running, int reason)

{

    SimpleSpiceDisplay *ssd = opaque;



    if (running) {

        ssd->worker->start(ssd->worker);

    } else {

        qemu_mutex_unlock_iothread();

        ssd->worker->stop(ssd->worker);

        qemu_mutex_lock_iothread();

    }

    ssd->running = running;

}
