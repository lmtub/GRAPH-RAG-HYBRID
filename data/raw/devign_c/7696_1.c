static void ioport_write(void *opaque, uint32_t addr, uint32_t val)

{

    PCIQXLDevice *d = opaque;

    uint32_t io_port = addr - d->io_base;



    switch (io_port) {

    case QXL_IO_RESET:

    case QXL_IO_SET_MODE:

    case QXL_IO_MEMSLOT_ADD:

    case QXL_IO_MEMSLOT_DEL:

    case QXL_IO_CREATE_PRIMARY:

        break;

    default:

        if (d->mode == QXL_MODE_NATIVE || d->mode == QXL_MODE_COMPAT)

            break;

        dprint(d, 1, "%s: unexpected port 0x%x in vga mode\n", __FUNCTION__, io_port);

        return;

    }



    switch (io_port) {

    case QXL_IO_UPDATE_AREA:

    {

        QXLRect update = d->ram->update_area;

        qemu_mutex_unlock_iothread();

        d->ssd.worker->update_area(d->ssd.worker, d->ram->update_surface,

                                   &update, NULL, 0, 0);

        qemu_mutex_lock_iothread();

        break;

    }

    case QXL_IO_NOTIFY_CMD:

        d->ssd.worker->wakeup(d->ssd.worker);

        break;

    case QXL_IO_NOTIFY_CURSOR:

        d->ssd.worker->wakeup(d->ssd.worker);

        break;

    case QXL_IO_UPDATE_IRQ:

        qxl_set_irq(d);

        break;

    case QXL_IO_NOTIFY_OOM:

        if (!SPICE_RING_IS_EMPTY(&d->ram->release_ring)) {

            break;

        }

        pthread_yield();

        if (!SPICE_RING_IS_EMPTY(&d->ram->release_ring)) {

            break;

        }

        d->oom_running = 1;

        d->ssd.worker->oom(d->ssd.worker);

        d->oom_running = 0;

        break;

    case QXL_IO_SET_MODE:

        dprint(d, 1, "QXL_SET_MODE %d\n", val);

        qxl_set_mode(d, val, 0);

        break;

    case QXL_IO_LOG:

        if (d->guestdebug) {

            fprintf(stderr, "qxl/guest: %s", d->ram->log_buf);

        }

        break;

    case QXL_IO_RESET:

        dprint(d, 1, "QXL_IO_RESET\n");

        qxl_hard_reset(d, 0);

        break;

    case QXL_IO_MEMSLOT_ADD:

        PANIC_ON(val >= NUM_MEMSLOTS);

        PANIC_ON(d->guest_slots[val].active);

        d->guest_slots[val].slot = d->ram->mem_slot;

        qxl_add_memslot(d, val, 0);

        break;

    case QXL_IO_MEMSLOT_DEL:

        qxl_del_memslot(d, val);

        break;

    case QXL_IO_CREATE_PRIMARY:

        PANIC_ON(val != 0);

        dprint(d, 1, "QXL_IO_CREATE_PRIMARY\n");

        d->guest_primary.surface = d->ram->create_surface;

        qxl_create_guest_primary(d, 0);

        break;

    case QXL_IO_DESTROY_PRIMARY:

        PANIC_ON(val != 0);

        dprint(d, 1, "QXL_IO_DESTROY_PRIMARY\n");

        qxl_destroy_primary(d);

        break;

    case QXL_IO_DESTROY_SURFACE_WAIT:

        d->ssd.worker->destroy_surface_wait(d->ssd.worker, val);

        break;

    case QXL_IO_DESTROY_ALL_SURFACES:

        d->ssd.worker->destroy_surfaces(d->ssd.worker);

        break;

    default:

        fprintf(stderr, "%s: ioport=0x%x, abort()\n", __FUNCTION__, io_port);

        abort();

    }

}
