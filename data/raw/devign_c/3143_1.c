static void vnc_connect(VncDisplay *vd, int csock, int skipauth)

{

    VncState *vs = g_malloc0(sizeof(VncState));

    int i;



    vs->csock = csock;



    if (skipauth) {

	vs->auth = VNC_AUTH_NONE;

#ifdef CONFIG_VNC_TLS

	vs->subauth = VNC_AUTH_INVALID;

#endif

    } else {

	vs->auth = vd->auth;

#ifdef CONFIG_VNC_TLS

	vs->subauth = vd->subauth;

#endif

    }



    vs->lossy_rect = g_malloc0(VNC_STAT_ROWS * sizeof (*vs->lossy_rect));

    for (i = 0; i < VNC_STAT_ROWS; ++i) {

        vs->lossy_rect[i] = g_malloc0(VNC_STAT_COLS * sizeof (uint8_t));

    }



    VNC_DEBUG("New client on socket %d\n", csock);

    dcl->idle = 0;

    socket_set_nonblock(vs->csock);

    qemu_set_fd_handler2(vs->csock, NULL, vnc_client_read, NULL, vs);



    vnc_client_cache_addr(vs);

    vnc_qmp_event(vs, QEVENT_VNC_CONNECTED);

    vnc_set_share_mode(vs, VNC_SHARE_MODE_CONNECTING);



    vs->vd = vd;

    vs->ds = vd->ds;

    vs->last_x = -1;

    vs->last_y = -1;



    vs->as.freq = 44100;

    vs->as.nchannels = 2;

    vs->as.fmt = AUD_FMT_S16;

    vs->as.endianness = 0;



#ifdef CONFIG_VNC_THREAD

    qemu_mutex_init(&vs->output_mutex);


#endif



    QTAILQ_INSERT_HEAD(&vd->clients, vs, next);



    vga_hw_update();



    vnc_write(vs, "RFB 003.008\n", 12);

    vnc_flush(vs);

    vnc_read_when(vs, protocol_version, 12);

    reset_keys(vs);

    if (vs->vd->lock_key_sync)

        vs->led = qemu_add_led_event_handler(kbd_leds, vs);



    vs->mouse_mode_notifier.notify = check_pointer_type_change;

    qemu_add_mouse_mode_change_notifier(&vs->mouse_mode_notifier);



    vnc_init_timer(vd);



    /* vs might be free()ed here */

}