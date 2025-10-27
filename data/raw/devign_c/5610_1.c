static void detach(sPAPRDRConnector *drc, DeviceState *d,
                   spapr_drc_detach_cb *detach_cb,
                   void *detach_cb_opaque, Error **errp)
{
    DPRINTFN("drc: %x, detach", get_index(drc));
    drc->detach_cb = detach_cb;
    drc->detach_cb_opaque = detach_cb_opaque;
    /* if we've signalled device presence to the guest, or if the guest
     * has gone ahead and configured the device (via manually-executed
     * device add via drmgr in guest, namely), we need to wait
     * for the guest to quiesce the device before completing detach.
     * Otherwise, we can assume the guest hasn't seen it and complete the
     * detach immediately. Note that there is a small race window
     * just before, or during, configuration, which is this context
     * refers mainly to fetching the device tree via RTAS.
     * During this window the device access will be arbitrated by
     * associated DRC, which will simply fail the RTAS calls as invalid.
     * This is recoverable within guest and current implementations of
     * drmgr should be able to cope.
     */
    if (!drc->signalled && !drc->configured) {
        /* if the guest hasn't seen the device we can't rely on it to
         * set it back to an isolated state via RTAS, so do it here manually
         */
        drc->isolation_state = SPAPR_DR_ISOLATION_STATE_ISOLATED;
    if (drc->isolation_state != SPAPR_DR_ISOLATION_STATE_ISOLATED) {
        DPRINTFN("awaiting transition to isolated state before removal");
    if (drc->type != SPAPR_DR_CONNECTOR_TYPE_PCI &&
        drc->allocation_state != SPAPR_DR_ALLOCATION_STATE_UNUSABLE) {
        DPRINTFN("awaiting transition to unusable state before removal");
    drc->indicator_state = SPAPR_DR_INDICATOR_STATE_INACTIVE;
    if (drc->detach_cb) {
        drc->detach_cb(drc->dev, drc->detach_cb_opaque);
    drc->awaiting_release = false;
    g_free(drc->fdt);
    drc->fdt = NULL;
    drc->fdt_start_offset = 0;
    object_property_del(OBJECT(drc), "device", NULL);
    drc->dev = NULL;
    drc->detach_cb = NULL;
    drc->detach_cb_opaque = NULL;