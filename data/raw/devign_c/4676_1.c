VirtIODevice *virtio_serial_init(DeviceState *dev, virtio_serial_conf *conf)

{

    VirtIOSerial *vser;

    VirtIODevice *vdev;

    uint32_t i, max_supported_ports;



    if (!conf->max_virtserial_ports)

        return NULL;



    /* Each port takes 2 queues, and one pair is for the control queue */

    max_supported_ports = VIRTIO_PCI_QUEUE_MAX / 2 - 1;



    if (conf->max_virtserial_ports > max_supported_ports) {

        error_report("maximum ports supported: %u", max_supported_ports);

        return NULL;

    }



    vdev = virtio_common_init("virtio-serial", VIRTIO_ID_CONSOLE,

                              sizeof(struct virtio_console_config),

                              sizeof(VirtIOSerial));



    vser = DO_UPCAST(VirtIOSerial, vdev, vdev);



    /* Spawn a new virtio-serial bus on which the ports will ride as devices */

    vser->bus = virtser_bus_new(dev);

    vser->bus->vser = vser;

    QTAILQ_INIT(&vser->ports);



    vser->bus->max_nr_ports = conf->max_virtserial_ports;

    vser->ivqs = qemu_malloc(conf->max_virtserial_ports * sizeof(VirtQueue *));

    vser->ovqs = qemu_malloc(conf->max_virtserial_ports * sizeof(VirtQueue *));



    /* Add a queue for host to guest transfers for port 0 (backward compat) */

    vser->ivqs[0] = virtio_add_queue(vdev, 128, handle_input);

    /* Add a queue for guest to host transfers for port 0 (backward compat) */

    vser->ovqs[0] = virtio_add_queue(vdev, 128, handle_output);



    /* TODO: host to guest notifications can get dropped

     * if the queue fills up. Implement queueing in host,

     * this might also make it possible to reduce the control

     * queue size: as guest preposts buffers there,

     * this will save 4Kbyte of guest memory per entry. */



    /* control queue: host to guest */

    vser->c_ivq = virtio_add_queue(vdev, 32, control_in);

    /* control queue: guest to host */

    vser->c_ovq = virtio_add_queue(vdev, 32, control_out);



    for (i = 1; i < vser->bus->max_nr_ports; i++) {

        /* Add a per-port queue for host to guest transfers */

        vser->ivqs[i] = virtio_add_queue(vdev, 128, handle_input);

        /* Add a per-per queue for guest to host transfers */

        vser->ovqs[i] = virtio_add_queue(vdev, 128, handle_output);

    }



    vser->config.max_nr_ports = tswap32(conf->max_virtserial_ports);

    vser->ports_map = qemu_mallocz(((conf->max_virtserial_ports + 31) / 32)

        * sizeof(vser->ports_map[0]));

    /*

     * Reserve location 0 for a console port for backward compat

     * (old kernel, new qemu)

     */

    mark_port_added(vser, 0);



    vser->vdev.get_features = get_features;

    vser->vdev.get_config = get_config;

    vser->vdev.set_config = set_config;



    vser->qdev = dev;



    /*

     * Register for the savevm section with the virtio-console name

     * to preserve backward compat

     */

    register_savevm(dev, "virtio-console", -1, 3, virtio_serial_save,

                    virtio_serial_load, vser);



    return vdev;

}
