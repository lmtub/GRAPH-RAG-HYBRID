void vnc_client_error(VncState *vs)

{

    vnc_client_io_error(vs, -1, EINVAL);

}
