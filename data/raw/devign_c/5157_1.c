static ssize_t flush_buf(VirtIOSerialPort *port,
                         const uint8_t *buf, ssize_t len)
{
    VirtConsole *vcon = VIRTIO_CONSOLE(port);
    ssize_t ret;
    if (!vcon->chr) {
        /* If there's no backend, we can just say we consumed all data. */
        return len;
    }
    ret = qemu_chr_fe_write(vcon->chr, buf, len);
    trace_virtio_console_flush_buf(port->id, len, ret);
    if (ret < len) {
        VirtIOSerialPortClass *k = VIRTIO_SERIAL_PORT_GET_CLASS(port);
        /*
         * Ideally we'd get a better error code than just -1, but
         * that's what the chardev interface gives us right now.  If
         * we had a finer-grained message, like -EPIPE, we could close
         * this connection.
        if (ret < 0)
            ret = 0;
        if (!k->is_console) {
            virtio_serial_throttle_port(port, true);
            if (!vcon->watch) {
                vcon->watch = qemu_chr_fe_add_watch(vcon->chr,
                                                    G_IO_OUT|G_IO_HUP,
                                                    chr_write_unblocked, vcon);
            }
        }
    }
    return ret;
}