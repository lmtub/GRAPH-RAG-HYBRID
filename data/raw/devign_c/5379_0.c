static void bt_hci_reset(struct bt_hci_s *hci)

{

    hci->acl_len = 0;

    hci->last_cmd = 0;

    hci->lm.connecting = 0;



    hci->event_mask[0] = 0xff;

    hci->event_mask[1] = 0xff;

    hci->event_mask[2] = 0xff;

    hci->event_mask[3] = 0xff;

    hci->event_mask[4] = 0xff;

    hci->event_mask[5] = 0x1f;

    hci->event_mask[6] = 0x00;

    hci->event_mask[7] = 0x00;

    hci->device.inquiry_scan = 0;

    hci->device.page_scan = 0;

    if (hci->device.lmp_name)

        g_free((void *) hci->device.lmp_name);

    hci->device.lmp_name = NULL;

    hci->device.class[0] = 0x00;

    hci->device.class[1] = 0x00;

    hci->device.class[2] = 0x00;

    hci->voice_setting = 0x0000;

    hci->conn_accept_tout = 0x1f40;

    hci->lm.inquiry_mode = 0x00;



    hci->psb_handle = 0x000;

    hci->asb_handle = 0x000;



    /* XXX: timer_del(sl->acl_mode_timer); for all links */

    timer_del(hci->lm.inquiry_done);

    timer_del(hci->lm.inquiry_next);

    timer_del(hci->conn_accept_timer);

}
