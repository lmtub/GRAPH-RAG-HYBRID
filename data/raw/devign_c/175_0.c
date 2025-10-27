static int virtser_port_qdev_init(DeviceState *qdev, DeviceInfo *base)

{

    VirtIOSerialPort *port = DO_UPCAST(VirtIOSerialPort, dev, qdev);

    VirtIOSerialPortInfo *info = DO_UPCAST(VirtIOSerialPortInfo, qdev, base);

    VirtIOSerialBus *bus = DO_UPCAST(VirtIOSerialBus, qbus, qdev->parent_bus);

    int ret, max_nr_ports;

    bool plugging_port0;



    port->vser = bus->vser;

    port->bh = qemu_bh_new(flush_queued_data_bh, port);



    /*

     * Is the first console port we're seeing? If so, put it up at

     * location 0. This is done for backward compatibility (old

     * kernel, new qemu).

     */

    plugging_port0 = port->is_console && !find_port_by_id(port->vser, 0);



    if (find_port_by_id(port->vser, port->id)) {

        error_report("virtio-serial-bus: A port already exists at id %u\n",

                     port->id);

        return -1;

    }



    if (port->id == VIRTIO_CONSOLE_BAD_ID) {

        if (plugging_port0) {

            port->id = 0;

        } else {

            port->id = find_free_port_id(port->vser);

            if (port->id == VIRTIO_CONSOLE_BAD_ID) {

                error_report("virtio-serial-bus: Maximum port limit for this device reached\n");

                return -1;

            }

        }

    }



    max_nr_ports = tswap32(port->vser->config.max_nr_ports);

    if (port->id >= max_nr_ports) {

        error_report("virtio-serial-bus: Out-of-range port id specified, max. allowed: %u\n",

                     max_nr_ports - 1);

        return -1;

    }



    port->info = info;

    ret = info->init(port);

    if (ret) {

        return ret;

    }



    if (!use_multiport(port->vser)) {

        /*

         * Allow writes to guest in this case; we have no way of

         * knowing if a guest port is connected.

         */

        port->guest_connected = true;

    }



    port->elem.out_num = 0;



    QTAILQ_INSERT_TAIL(&port->vser->ports, port, next);

    port->ivq = port->vser->ivqs[port->id];

    port->ovq = port->vser->ovqs[port->id];



    add_port(port->vser, port->id);



    /* Send an update to the guest about this new port added */

    virtio_notify_config(&port->vser->vdev);



    return ret;

}
