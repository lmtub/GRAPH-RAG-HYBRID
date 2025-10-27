static void vnc_connect(VncDisplay *vd, int csock)

{

    VncState *vs = qemu_mallocz(sizeof(VncState));

    vs->csock = csock;



    VNC_DEBUG("New client on socket %d\n", csock);

    dcl->idle = 0;

    socket_set_nonblock(vs->csock);

    qemu_set_fd_handler2(vs->csock, NULL, vnc_client_read, NULL, vs);



    vs->vd = vd;

    vs->ds = vd->ds;

    vs->timer = qemu_new_timer(rt_clock, vnc_update_client, vs);

    vs->last_x = -1;

    vs->last_y = -1;



    vs->as.freq = 44100;

    vs->as.nchannels = 2;

    vs->as.fmt = AUD_FMT_S16;

    vs->as.endianness = 0;



    vnc_resize(vs);

    vnc_write(vs, "RFB 003.008\n", 12);

    vnc_flush(vs);

    vnc_read_when(vs, protocol_version, 12);

    memset(vs->old_data, 0, ds_get_linesize(vs->ds) * ds_get_height(vs->ds));

    memset(vs->dirty_row, 0xFF, sizeof(vs->dirty_row));

    vnc_update_client(vs);

    reset_keys(vs);



    vs->next = vd->clients;

    vd->clients = vs;

}
