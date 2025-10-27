vcard_emul_login(VCard *card, unsigned char *pin, int pin_len)

{

    PK11SlotInfo *slot;

    unsigned char *pin_string = NULL;

    int i;

    SECStatus rv;



    if (!nss_emul_init) {

        return VCARD7816_STATUS_ERROR_CONDITION_NOT_SATISFIED;

    }

    slot = vcard_emul_card_get_slot(card);

     /* We depend on the PKCS #11 module internal login state here because we

      * create a separate process to handle each guest instance. If we needed

      * to handle multiple guests from one process, then we would need to keep

      * a lot of extra state in our card structure

      * */

    pin_string = g_malloc(pin_len+1);

    memcpy(pin_string, pin, pin_len);

    pin_string[pin_len] = 0;



    /* handle CAC expanded pins correctly */

    for (i = pin_len-1; i >= 0 && (pin_string[i] == 0xff); i--) {

        pin_string[i] = 0;

    }



    rv = PK11_Authenticate(slot, PR_FALSE, pin_string);

    memset(pin_string, 0, pin_len);  /* don't let the pin hang around in memory

                                        to be snooped */

    g_free(pin_string);

    if (rv == SECSuccess) {

        return VCARD7816_STATUS_SUCCESS;

    }

    /* map the error from port get error */

    return VCARD7816_STATUS_ERROR_CONDITION_NOT_SATISFIED;

}
