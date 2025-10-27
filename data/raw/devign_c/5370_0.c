static ssize_t flush_buf(VirtIOSerialPort *port, const uint8_t *buf, size_t len)

{

    VirtConsole *vcon = DO_UPCAST(VirtConsole, port, port);

    ssize_t ret;



    ret = qemu_chr_write(vcon->chr, buf, len);



    trace_virtio_console_flush_buf(port->id, len, ret);

    return ret;

}
