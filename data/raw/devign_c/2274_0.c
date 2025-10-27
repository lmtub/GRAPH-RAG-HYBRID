static void set_guest_connected(VirtIOSerialPort *port, int guest_connected)

{

    VirtConsole *vcon = VIRTIO_CONSOLE(port);



    if (!vcon->chr) {

        return;

    }

    qemu_chr_fe_set_open(vcon->chr, guest_connected);

}
