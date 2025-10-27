void vnc_write(VncState *vs, const void *data, size_t len)

{

    buffer_reserve(&vs->output, len);



    if (buffer_empty(&vs->output)) {

        qemu_set_fd_handler2(vs->csock, NULL, vnc_client_read, vnc_client_write, vs);

    }



    buffer_append(&vs->output, data, len);

}
