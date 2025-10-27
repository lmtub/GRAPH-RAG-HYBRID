static void csrhci_in_packet_vendor(struct csrhci_s *s, int ocf,

                uint8_t *data, int len)

{

    int offset;

    uint8_t *rpkt;



    switch (ocf) {

    case OCF_CSR_SEND_FIRMWARE:

        /* Check if this is the bd_address packet */

        if (len >= 18 + 8 && data[12] == 0x01 && data[13] == 0x00) {

            offset = 18;

            s->bd_addr.b[0] = data[offset + 7];	/* Beyond cmd packet end(!?) */

            s->bd_addr.b[1] = data[offset + 6];

            s->bd_addr.b[2] = data[offset + 4];

            s->bd_addr.b[3] = data[offset + 0];

            s->bd_addr.b[4] = data[offset + 3];

            s->bd_addr.b[5] = data[offset + 2];



            s->hci->bdaddr_set(s->hci, s->bd_addr.b);

            fprintf(stderr, "%s: bd_address loaded from firmware: "

                            "%02x:%02x:%02x:%02x:%02x:%02x\n", __FUNCTION__,

                            s->bd_addr.b[0], s->bd_addr.b[1], s->bd_addr.b[2],

                            s->bd_addr.b[3], s->bd_addr.b[4], s->bd_addr.b[5]);

        }



        rpkt = csrhci_out_packet_event(s, EVT_VENDOR, 11);

        /* Status bytes: no error */

        rpkt[9] = 0x00;

        rpkt[10] = 0x00;

        break;



    default:

        fprintf(stderr, "%s: got a bad CMD packet\n", __FUNCTION__);

        return;

    }



    csrhci_fifo_wake(s);

}
