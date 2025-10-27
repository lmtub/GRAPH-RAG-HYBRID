static int usbnet_can_receive(VLANClientState *nc)

{

    USBNetState *s = DO_UPCAST(NICState, nc, nc)->opaque;



    if (is_rndis(s) && !s->rndis_state == RNDIS_DATA_INITIALIZED) {

        return 1;

    }



    return !s->in_len;

}
