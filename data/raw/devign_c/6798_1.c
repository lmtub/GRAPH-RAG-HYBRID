char *vnc_display_local_addr(DisplayState *ds)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;

    

    return vnc_socket_local_addr("%s:%s", vs->lsock);

}
