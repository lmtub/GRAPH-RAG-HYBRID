static void submit_pdu(V9fsState *s, V9fsPDU *pdu)

{

    pdu_handler_t *handler;



    if (debug_9p_pdu) {

        pprint_pdu(pdu);

    }



    BUG_ON(pdu->id >= ARRAY_SIZE(pdu_handlers));



    handler = pdu_handlers[pdu->id];

    BUG_ON(handler == NULL);



    handler(s, pdu);

}
