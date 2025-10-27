static void vnc_set_share_mode(VncState *vs, VncShareMode mode)

{

#ifdef _VNC_DEBUG

    static const char *mn[] = {

        [0]                           = "undefined",

        [VNC_SHARE_MODE_CONNECTING]   = "connecting",

        [VNC_SHARE_MODE_SHARED]       = "shared",

        [VNC_SHARE_MODE_EXCLUSIVE]    = "exclusive",

        [VNC_SHARE_MODE_DISCONNECTED] = "disconnected",

    };

    fprintf(stderr, "%s/%d: %s -> %s\n", __func__,

            vs->csock, mn[vs->share_mode], mn[mode]);

#endif



    if (vs->share_mode == VNC_SHARE_MODE_EXCLUSIVE) {

        vs->vd->num_exclusive--;

    }

    vs->share_mode = mode;

    if (vs->share_mode == VNC_SHARE_MODE_EXCLUSIVE) {

        vs->vd->num_exclusive++;

    }

}
