static void ehci_reset(void *opaque)

{

    EHCIState *s = opaque;

    int i;

    USBDevice *devs[NB_PORTS];



    trace_usb_ehci_reset();



    /*

     * Do the detach before touching portsc, so that it correctly gets send to

     * us or to our companion based on PORTSC_POWNER before the reset.

     */

    for(i = 0; i < NB_PORTS; i++) {

        devs[i] = s->ports[i].dev;

        if (devs[i] && devs[i]->attached) {

            usb_detach(&s->ports[i]);

        }

    }



    memset(&s->mmio[OPREGBASE], 0x00, MMIO_SIZE - OPREGBASE);



    s->usbcmd = NB_MAXINTRATE << USBCMD_ITC_SH;

    s->usbsts = USBSTS_HALT;

    s->usbsts_pending = 0;

    s->usbsts_frindex = 0;



    s->astate = EST_INACTIVE;

    s->pstate = EST_INACTIVE;



    for(i = 0; i < NB_PORTS; i++) {

        if (s->companion_ports[i]) {

            s->portsc[i] = PORTSC_POWNER | PORTSC_PPOWER;

        } else {

            s->portsc[i] = PORTSC_PPOWER;

        }

        if (devs[i] && devs[i]->attached) {

            usb_attach(&s->ports[i]);

            usb_device_reset(devs[i]);

        }

    }

    ehci_queues_rip_all(s, 0);

    ehci_queues_rip_all(s, 1);

    qemu_del_timer(s->frame_timer);

    qemu_bh_cancel(s->async_bh);

}
