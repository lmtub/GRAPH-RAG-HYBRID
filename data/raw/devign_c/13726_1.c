char *vnc_display_local_addr(const char *id)

{

    VncDisplay *vs = vnc_display_find(id);




    return vnc_socket_local_addr("%s:%s", vs->lsock);

}