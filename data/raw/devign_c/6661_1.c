static void virtconsole_realize(DeviceState *dev, Error **errp)

{

    VirtIOSerialPort *port = VIRTIO_SERIAL_PORT(dev);

    VirtConsole *vcon = VIRTIO_CONSOLE(dev);

    VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_GET_CLASS(dev);



    if (port->id == 0 && !k->is_console) {

        error_setg(errp, "Port number 0 on virtio-serial devices reserved "

                   "for virtconsole devices for backward compatibility.");

        return;

    }



    if (vcon->chr) {

        vcon->chr->explicit_fe_open = 1;

        qemu_chr_add_handlers(vcon->chr, chr_can_read, chr_read, chr_event,

                              vcon);

    }

}
