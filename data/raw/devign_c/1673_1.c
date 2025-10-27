static void v9fs_flush(void *opaque)

{

    int16_t tag;

    size_t offset = 7;

    V9fsPDU *cancel_pdu;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "w", &tag);




    QLIST_FOREACH(cancel_pdu, &s->active_list, next) {

        if (cancel_pdu->tag == tag) {

            break;

        }

    }

    if (cancel_pdu) {

        cancel_pdu->cancelled = 1;

        /*

         * Wait for pdu to complete.

         */

        qemu_co_queue_wait(&cancel_pdu->complete);

        cancel_pdu->cancelled = 0;

        free_pdu(pdu->s, cancel_pdu);

    }

    complete_pdu(s, pdu, 7);

    return;

}