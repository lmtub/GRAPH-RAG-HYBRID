static int uhci_complete_td(UHCIState *s, UHCI_TD *td, UHCIAsync *async, uint32_t *int_mask)

{

    int len = 0, max_len, err, ret;

    uint8_t pid;



    max_len = ((td->token >> 21) + 1) & 0x7ff;

    pid = td->token & 0xff;



    ret = async->packet.result;



    if (td->ctrl & TD_CTRL_IOS)

        td->ctrl &= ~TD_CTRL_ACTIVE;



    if (ret < 0)

        goto out;



    len = async->packet.result;

    td->ctrl = (td->ctrl & ~0x7ff) | ((len - 1) & 0x7ff);



    /* The NAK bit may have been set by a previous frame, so clear it

       here.  The docs are somewhat unclear, but win2k relies on this

       behavior.  */

    td->ctrl &= ~(TD_CTRL_ACTIVE | TD_CTRL_NAK);

    if (td->ctrl & TD_CTRL_IOC)

        *int_mask |= 0x01;



    if (pid == USB_TOKEN_IN) {

        if (len > max_len) {

            ret = USB_RET_BABBLE;

            goto out;

        }



        if ((td->ctrl & TD_CTRL_SPD) && len < max_len) {

            *int_mask |= 0x02;

            /* short packet: do not update QH */

            trace_usb_uhci_packet_complete_shortxfer(async->queue->token,

                                                    async->td);

            return TD_RESULT_NEXT_QH;

        }

    }



    /* success */

    trace_usb_uhci_packet_complete_success(async->queue->token, async->td);

    return TD_RESULT_COMPLETE;



out:

    /*

     * We should not do any further processing on a queue with errors!

     * This is esp. important for bulk endpoints with pipelining enabled

     * (redirection to a real USB device), where we must cancel all the

     * transfers after this one so that:

     * 1) If they've completed already, they are not processed further

     *    causing more stalls, originating from the same failed transfer

     * 2) If still in flight, they are cancelled before the guest does

     *    a clear stall, otherwise the guest and device can loose sync!

     */

    while (!QTAILQ_EMPTY(&async->queue->asyncs)) {

        UHCIAsync *as = QTAILQ_FIRST(&async->queue->asyncs);

        uhci_async_unlink(as);

        uhci_async_cancel(as);

    }



    switch(ret) {

    case USB_RET_STALL:

        td->ctrl |= TD_CTRL_STALL;

        td->ctrl &= ~TD_CTRL_ACTIVE;

        s->status |= UHCI_STS_USBERR;

        if (td->ctrl & TD_CTRL_IOC) {

            *int_mask |= 0x01;

        }

        uhci_update_irq(s);

        trace_usb_uhci_packet_complete_stall(async->queue->token, async->td);

        return TD_RESULT_NEXT_QH;



    case USB_RET_BABBLE:

        td->ctrl |= TD_CTRL_BABBLE | TD_CTRL_STALL;

        td->ctrl &= ~TD_CTRL_ACTIVE;

        s->status |= UHCI_STS_USBERR;

        if (td->ctrl & TD_CTRL_IOC) {

            *int_mask |= 0x01;

        }

        uhci_update_irq(s);

        /* frame interrupted */

        trace_usb_uhci_packet_complete_babble(async->queue->token, async->td);

        return TD_RESULT_STOP_FRAME;



    case USB_RET_NAK:

        td->ctrl |= TD_CTRL_NAK;

        if (pid == USB_TOKEN_SETUP)

            break;

        return TD_RESULT_NEXT_QH;



    case USB_RET_IOERROR:

    case USB_RET_NODEV:

    default:

	break;

    }



    /* Retry the TD if error count is not zero */



    td->ctrl |= TD_CTRL_TIMEOUT;

    err = (td->ctrl >> TD_CTRL_ERROR_SHIFT) & 3;

    if (err != 0) {

        err--;

        if (err == 0) {

            td->ctrl &= ~TD_CTRL_ACTIVE;

            s->status |= UHCI_STS_USBERR;

            if (td->ctrl & TD_CTRL_IOC)

                *int_mask |= 0x01;

            uhci_update_irq(s);

            trace_usb_uhci_packet_complete_error(async->queue->token,

                                                 async->td);

        }

    }

    td->ctrl = (td->ctrl & ~(3 << TD_CTRL_ERROR_SHIFT)) |

        (err << TD_CTRL_ERROR_SHIFT);

    return TD_RESULT_NEXT_QH;

}
