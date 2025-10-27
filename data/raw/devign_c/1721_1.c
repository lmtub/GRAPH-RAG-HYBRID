static inline void bt_hci_event_complete_read_local_name(struct bt_hci_s *hci)

{

    read_local_name_rp params;

    params.status = HCI_SUCCESS;

    memset(params.name, 0, sizeof(params.name));

    if (hci->device.lmp_name)

        strncpy(params.name, hci->device.lmp_name, sizeof(params.name));



    bt_hci_event_complete(hci, &params, READ_LOCAL_NAME_RP_SIZE);

}
