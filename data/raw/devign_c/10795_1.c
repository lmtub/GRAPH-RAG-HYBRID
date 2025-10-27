static void uhci_frame_timer(void *opaque)

{

    UHCIState *s = opaque;



    /* prepare the timer for the next frame */

    s->expire_time += (get_ticks_per_sec() / FRAME_TIMER_FREQ);

    s->frame_bytes = 0;

    s->completions_only = false;

    qemu_bh_cancel(s->bh);



    if (!(s->cmd & UHCI_CMD_RS)) {

        /* Full stop */

        trace_usb_uhci_schedule_stop();

        qemu_del_timer(s->frame_timer);

        uhci_async_cancel_all(s);

        /* set hchalted bit in status - UHCI11D 2.1.2 */

        s->status |= UHCI_STS_HCHALTED;

        return;

    }



    /* Process the current frame */

    trace_usb_uhci_frame_start(s->frnum);



    uhci_async_validate_begin(s);



    uhci_process_frame(s);



    uhci_async_validate_end(s);



    /* The uhci spec says frnum reflects the frame currently being processed,

     * and the guest must look at frnum - 1 on interrupt, so inc frnum now */

    s->frnum = (s->frnum + 1) & 0x7ff;



    /* Complete the previous frame */

    if (s->pending_int_mask) {

        s->status2 |= s->pending_int_mask;

        s->status  |= UHCI_STS_USBINT;

        uhci_update_irq(s);

    }

    s->pending_int_mask = 0;



    qemu_mod_timer(s->frame_timer, s->expire_time);

}
