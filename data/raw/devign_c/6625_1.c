void vnc_flush(VncState *vs)

{

    if (vs->output.offset)

        vnc_client_write(vs);

}
