void qdev_free(DeviceState *dev)

{

    BusState *bus;




    if (dev->state == DEV_STATE_INITIALIZED) {

        while (dev->num_child_bus) {

            bus = QLIST_FIRST(&dev->child_bus);

            qbus_free(bus);

        }

        if (dev->info->vmsd)

            vmstate_unregister(dev->info->vmsd, dev);

        if (dev->info->exit)

            dev->info->exit(dev);

        if (dev->opts)

            qemu_opts_del(dev->opts);

    }

    qemu_unregister_reset(qdev_reset, dev);

    QLIST_REMOVE(dev, sibling);

    for (prop = dev->info->props; prop && prop->name; prop++) {

        if (prop->info->free) {

            prop->info->free(dev, prop);

        }

    }

    qemu_free(dev);

}