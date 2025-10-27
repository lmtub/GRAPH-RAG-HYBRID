static int ccid_initfn(USBDevice *dev)

{

    USBCCIDState *s = DO_UPCAST(USBCCIDState, dev, dev);



    s->bus = ccid_bus_new(&dev->qdev);

    s->card = NULL;

    s->cardinfo = NULL;

    s->migration_state = MIGRATION_NONE;

    s->migration_target_ip = 0;

    s->migration_target_port = 0;

    s->dev.speed = USB_SPEED_FULL;

    s->notify_slot_change = false;

    s->powered = true;

    s->pending_answers_num = 0;

    s->last_answer_error = 0;

    s->bulk_in_pending_start = 0;

    s->bulk_in_pending_end = 0;

    s->current_bulk_in = NULL;

    ccid_reset_error_status(s);

    s->bulk_out_pos = 0;

    ccid_reset_parameters(s);

    ccid_reset(s);

    return 0;

}
