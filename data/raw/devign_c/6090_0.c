static void virtio_serial_save_device(VirtIODevice *vdev, QEMUFile *f)

{

    VirtIOSerial *s = VIRTIO_SERIAL(vdev);

    VirtIOSerialPort *port;

    uint32_t nr_active_ports;

    unsigned int i, max_nr_ports;

    struct virtio_console_config config;



    /* The config space (ignored on the far end in current versions) */

    get_config(vdev, (uint8_t *)&config);

    qemu_put_be16s(f, &config.cols);

    qemu_put_be16s(f, &config.rows);

    qemu_put_be32s(f, &config.max_nr_ports);



    /* The ports map */

    max_nr_ports = s->serial.max_virtserial_ports;

    for (i = 0; i < (max_nr_ports + 31) / 32; i++) {

        qemu_put_be32s(f, &s->ports_map[i]);

    }



    /* Ports */



    nr_active_ports = 0;

    QTAILQ_FOREACH(port, &s->ports, next) {

        nr_active_ports++;

    }



    qemu_put_be32s(f, &nr_active_ports);



    /*

     * Items in struct VirtIOSerialPort.

     */

    QTAILQ_FOREACH(port, &s->ports, next) {

        uint32_t elem_popped;



        qemu_put_be32s(f, &port->id);

        qemu_put_byte(f, port->guest_connected);

        qemu_put_byte(f, port->host_connected);



	elem_popped = 0;

        if (port->elem.out_num) {

            elem_popped = 1;

        }

        qemu_put_be32s(f, &elem_popped);

        if (elem_popped) {

            qemu_put_be32s(f, &port->iov_idx);

            qemu_put_be64s(f, &port->iov_offset);



            qemu_put_buffer(f, (unsigned char *)&port->elem,

                            sizeof(port->elem));

        }

    }

}
