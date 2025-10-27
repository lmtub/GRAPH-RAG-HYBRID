static int vnc_display_disable_login(DisplayState *ds)

{

    VncDisplay *vs = vnc_display;



    if (!vs) {

        return -1;

    }



    if (vs->password) {

        g_free(vs->password);

    }



    vs->password = NULL;

    if (vs->auth == VNC_AUTH_NONE) {

        vs->auth = VNC_AUTH_VNC;

    }



    return 0;

}
