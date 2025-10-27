cac_applet_pki_reset(VCard *card, int channel)

{

    VCardAppletPrivate *applet_private = NULL;

    CACPKIAppletData *pki_applet = NULL;

    applet_private = vcard_get_current_applet_private(card, channel);

    assert(applet_private);

    pki_applet = &(applet_private->u.pki_data);



    pki_applet->cert_buffer = NULL;

    if (pki_applet->sign_buffer) {

        g_free(pki_applet->sign_buffer);

        pki_applet->sign_buffer = NULL;

    }

    pki_applet->cert_buffer_len = 0;

    pki_applet->sign_buffer_len = 0;

    return VCARD_DONE;

}
