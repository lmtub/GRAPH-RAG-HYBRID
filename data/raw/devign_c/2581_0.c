static void csrhci_in_packet(struct csrhci_s *s, uint8_t *pkt)

{

    uint8_t *rpkt;

    int opc;



    switch (*pkt ++) {

    case H4_CMD_PKT:

        opc = le16_to_cpu(((struct hci_command_hdr *) pkt)->opcode);

        if (cmd_opcode_ogf(opc) == OGF_VENDOR_CMD) {

            csrhci_in_packet_vendor(s, cmd_opcode_ocf(opc),

                            pkt + sizeof(struct hci_command_hdr),

                            s->in_len - sizeof(struct hci_command_hdr) - 1);

            return;

        }



        /* TODO: if the command is OCF_READ_LOCAL_COMMANDS or the likes,

         * we need to send it to the HCI layer and then add our supported

         * commands to the returned mask (such as OGF_VENDOR_CMD).  With

         * bt-hci.c we could just have hooks for this kind of commands but

         * we can't with bt-host.c.  */



        s->hci->cmd_send(s->hci, pkt, s->in_len - 1);

        break;



    case H4_EVT_PKT:

        goto bad_pkt;



    case H4_ACL_PKT:

        s->hci->acl_send(s->hci, pkt, s->in_len - 1);

        break;



    case H4_SCO_PKT:

        s->hci->sco_send(s->hci, pkt, s->in_len - 1);

        break;



    case H4_NEG_PKT:

        if (s->in_hdr != sizeof(csrhci_neg_packet) ||

                        memcmp(pkt - 1, csrhci_neg_packet, s->in_hdr)) {

            fprintf(stderr, "%s: got a bad NEG packet\n", __func__);

            return;

        }

        pkt += 2;



        rpkt = csrhci_out_packet_csr(s, H4_NEG_PKT, 10);



        *rpkt ++ = 0x20;	/* Operational settings negotiation Ok */

        memcpy(rpkt, pkt, 7); rpkt += 7;

        *rpkt ++ = 0xff;

        *rpkt = 0xff;

        break;



    case H4_ALIVE_PKT:

        if (s->in_hdr != 4 || pkt[1] != 0x55 || pkt[2] != 0x00) {

            fprintf(stderr, "%s: got a bad ALIVE packet\n", __func__);

            return;

        }



        rpkt = csrhci_out_packet_csr(s, H4_ALIVE_PKT, 2);



        *rpkt ++ = 0xcc;

        *rpkt = 0x00;

        break;



    default:

    bad_pkt:

        /* TODO: error out */

        fprintf(stderr, "%s: got a bad packet\n", __func__);

        break;

    }



    csrhci_fifo_wake(s);

}
