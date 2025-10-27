static int uhci_handle_td(UHCIState *s, UHCIQueue *q, uint32_t qh_addr,
                          UHCI_TD *td, uint32_t td_addr, uint32_t *int_mask)
{
    int len = 0, max_len;
    bool spd;
    bool queuing = (q != NULL);
    uint8_t pid = td->token & 0xff;
    UHCIAsync *async = uhci_async_find_td(s, td_addr);
    if (async) {
        if (uhci_queue_verify(async->queue, qh_addr, td, td_addr, queuing)) {
            assert(q == NULL || q == async->queue);
            q = async->queue;
        } else {
            uhci_queue_free(async->queue, "guest re-used pending td");
            async = NULL;
    if (q == NULL) {
        q = uhci_queue_find(s, td);
        if (q && !uhci_queue_verify(q, qh_addr, td, td_addr, queuing)) {
            uhci_queue_free(q, "guest re-used qh");
            q = NULL;
    if (q) {
        q->valid = 32;
    /* Is active ? */
    if (!(td->ctrl & TD_CTRL_ACTIVE)) {
        if (async) {
            /* Guest marked a pending td non-active, cancel the queue */
            uhci_queue_free(async->queue, "pending td non-active");
        /*
         * ehci11d spec page 22: "Even if the Active bit in the TD is already
         * cleared when the TD is fetched ... an IOC interrupt is generated"
         */
        if (td->ctrl & TD_CTRL_IOC) {
                *int_mask |= 0x01;
        return TD_RESULT_NEXT_QH;
    if (async) {
        if (queuing) {
            /* we are busy filling the queue, we are not prepared
               to consume completed packages then, just leave them
               in async state */
            return TD_RESULT_ASYNC_CONT;
        if (!async->done) {
            UHCI_TD last_td;
            UHCIAsync *last = QTAILQ_LAST(&async->queue->asyncs, asyncs_head);
            /*
             * While we are waiting for the current td to complete, the guest
             * may have added more tds to the queue. Note we re-read the td
             * rather then caching it, as we want to see guest made changes!
             */
            uhci_read_td(s, &last_td, last->td_addr);
            uhci_queue_fill(async->queue, &last_td);
            return TD_RESULT_ASYNC_CONT;
        uhci_async_unlink(async);
        goto done;
    /* Allocate new packet */
    if (q == NULL) {
        USBDevice *dev = uhci_find_device(s, (td->token >> 8) & 0x7f);
        USBEndpoint *ep = usb_ep_get(dev, pid, (td->token >> 15) & 0xf);
        q = uhci_queue_new(s, qh_addr, td, ep);
    async = uhci_async_alloc(q, td_addr);
    max_len = ((td->token >> 21) + 1) & 0x7ff;
    spd = (pid == USB_TOKEN_IN && (td->ctrl & TD_CTRL_SPD) != 0);
    usb_packet_setup(&async->packet, pid, q->ep, td_addr, spd,
                     (td->ctrl & TD_CTRL_IOC) != 0);
    qemu_sglist_add(&async->sgl, td->buffer, max_len);
    usb_packet_map(&async->packet, &async->sgl);
    switch(pid) {
    case USB_TOKEN_OUT:
    case USB_TOKEN_SETUP:
        len = usb_handle_packet(q->ep->dev, &async->packet);
        if (len >= 0)
            len = max_len;
        break;
    case USB_TOKEN_IN:
        len = usb_handle_packet(q->ep->dev, &async->packet);
        break;
    default:
        /* invalid pid : frame interrupted */
        usb_packet_unmap(&async->packet, &async->sgl);
        uhci_async_free(async);
        s->status |= UHCI_STS_HCPERR;
        uhci_update_irq(s);
        return TD_RESULT_STOP_FRAME;
    if (len == USB_RET_ASYNC) {
        uhci_async_link(async);
        if (!queuing) {
            uhci_queue_fill(q, td);
        return TD_RESULT_ASYNC_START;
    async->packet.result = len;
done:
    len = uhci_complete_td(s, td, async, int_mask);
    usb_packet_unmap(&async->packet, &async->sgl);
    uhci_async_free(async);
    return len;