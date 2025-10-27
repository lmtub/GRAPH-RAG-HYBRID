cac_delete_pki_applet_private(VCardAppletPrivate *applet_private)

{

    CACPKIAppletData *pki_applet_data = NULL;



    if (applet_private == NULL) {

        return;

    }

    pki_applet_data = &(applet_private->u.pki_data);

    if (pki_applet_data->cert != NULL) {

        g_free(pki_applet_data->cert);

    }

    if (pki_applet_data->sign_buffer != NULL) {

        g_free(pki_applet_data->sign_buffer);

    }

    if (pki_applet_data->key != NULL) {

        vcard_emul_delete_key(pki_applet_data->key);

    }

    g_free(applet_private);

}
