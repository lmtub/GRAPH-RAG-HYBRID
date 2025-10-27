int vnc_display_password(DisplayState *ds, const char *password)

{

    VncDisplay *vs = vnc_display;



    if (!vs) {

        return -EINVAL;

    }



    if (!password) {

        /* This is not the intention of this interface but err on the side

           of being safe */

        return vnc_display_disable_login(ds);

    }



    if (vs->password) {

        g_free(vs->password);

        vs->password = NULL;

    }

    vs->password = g_strdup(password);

    if (vs->auth == VNC_AUTH_NONE) {

        vs->auth = VNC_AUTH_VNC;

    }



    return 0;

}
