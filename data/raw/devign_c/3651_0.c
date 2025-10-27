static inline uint8_t *bt_hci_event_start(struct bt_hci_s *hci,

                int evt, int len)

{

    uint8_t *packet, mask;

    int mask_byte;



    if (len > 255) {

        fprintf(stderr, "%s: HCI event params too long (%ib)\n",

                        __FUNCTION__, len);

        exit(-1);

    }



    mask_byte = (evt - 1) >> 3;

    mask = 1 << ((evt - 1) & 3);

    if (mask & bt_event_reserved_mask[mask_byte] & ~hci->event_mask[mask_byte])

        return NULL;



    packet = hci->evt_packet(hci->opaque);

    packet[0] = evt;

    packet[1] = len;



    return &packet[2];

}
