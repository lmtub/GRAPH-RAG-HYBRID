int vnc_display_disable_login(DisplayState *ds)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;



    if (!vs) {

        return -1;

    }



    if (vs->password) {

        g_free(vs->password);

    }



    vs->password = NULL;

    vs->auth = VNC_AUTH_VNC;



    return 0;

}
