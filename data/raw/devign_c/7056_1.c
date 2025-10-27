static void vnc_connect(VncDisplay *vd, int csock, int skipauth, bool websocket)

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

#ifdef CONFIG_VNC_WS

    if (websocket) {

        vs->websocket = 1;

        qemu_set_fd_handler2(vs->csock, NULL, vncws_handshake_read, NULL, vs);

    } else

#endif /* CONFIG_VNC_WS */

    {

        qemu_set_fd_handler2(vs->csock, NULL, vnc_client_read, NULL, vs);

    }



    vnc_client_cache_addr(vs);

    vnc_qmp_event(vs, QEVENT_VNC_CONNECTED);

    vnc_set_share_mode(vs, VNC_SHARE_MODE_CONNECTING);



    vs->vd = vd;



#ifdef CONFIG_VNC_WS

    if (!vs->websocket)

#endif

    {

        vnc_init_state(vs);

    }

}
