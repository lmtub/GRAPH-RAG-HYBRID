static int vnc_display_listen(VncDisplay *vd,

                              SocketAddressLegacy **saddr,

                              size_t nsaddr,

                              SocketAddressLegacy **wsaddr,

                              size_t nwsaddr,

                              Error **errp)

{

    size_t i;



    for (i = 0; i < nsaddr; i++) {

        if (vnc_display_listen_addr(vd, saddr[i],

                                    "vnc-listen",

                                    &vd->lsock,

                                    &vd->lsock_tag,

                                    &vd->nlsock,

                                    errp) < 0) {

            return -1;

        }

    }

    for (i = 0; i < nwsaddr; i++) {

        if (vnc_display_listen_addr(vd, wsaddr[i],

                                    "vnc-ws-listen",

                                    &vd->lwebsock,

                                    &vd->lwebsock_tag,

                                    &vd->nlwebsock,

                                    errp) < 0) {

            return -1;

        }

    }



    return 0;

}
