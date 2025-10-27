vreader_xfr_bytes(VReader *reader,

                  unsigned char *send_buf, int send_buf_len,

                  unsigned char *receive_buf, int *receive_buf_len)

{

    VCardAPDU *apdu;

    VCardResponse *response = NULL;

    VCardStatus card_status;

    unsigned short status;

    VCard *card = vreader_get_card(reader);



    if (card == NULL) {

        return VREADER_NO_CARD;

    }



    apdu = vcard_apdu_new(send_buf, send_buf_len, &status);

    if (apdu == NULL) {

        response = vcard_make_response(status);

        card_status = VCARD_DONE;

    } else {

        g_debug("%s: CLS=0x%x,INS=0x%x,P1=0x%x,P2=0x%x,Lc=%d,Le=%d %s",

              __func__, apdu->a_cla, apdu->a_ins, apdu->a_p1, apdu->a_p2,

              apdu->a_Lc, apdu->a_Le, apdu_ins_to_string(apdu->a_ins));

        card_status = vcard_process_apdu(card, apdu, &response);

        if (response) {

            g_debug("%s: status=%d sw1=0x%x sw2=0x%x len=%d (total=%d)",

                  __func__, response->b_status, response->b_sw1,

                  response->b_sw2, response->b_len, response->b_total_len);

        }

    }

    assert(card_status == VCARD_DONE);

    if (card_status == VCARD_DONE) {

        int size = MIN(*receive_buf_len, response->b_total_len);

        memcpy(receive_buf, response->b_data, size);

        *receive_buf_len = size;

    }

    vcard_response_delete(response);

    vcard_apdu_delete(apdu);

    vcard_free(card); /* free our reference */

    return VREADER_OK;

}
