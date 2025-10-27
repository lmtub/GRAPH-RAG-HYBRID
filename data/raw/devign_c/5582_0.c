vcard_emul_mirror_card(VReader *vreader)

{

    /*

     * lookup certs using the C_FindObjects. The Stan Cert handle won't give

     * us the real certs until we log in.

     */

    PK11GenericObject *firstObj, *thisObj;

    int cert_count;

    unsigned char **certs;

    int *cert_len;

    VCardKey **keys;

    PK11SlotInfo *slot;

    VCard *card;



    slot = vcard_emul_reader_get_slot(vreader);

    if (slot == NULL) {

        return NULL;

    }



    firstObj = PK11_FindGenericObjects(slot, CKO_CERTIFICATE);

    if (firstObj == NULL) {

        return NULL;

    }



    /* count the certs */

    cert_count = 0;

    for (thisObj = firstObj; thisObj;

                             thisObj = PK11_GetNextGenericObject(thisObj)) {

        cert_count++;

    }



    if (cert_count == 0) {

        PK11_DestroyGenericObjects(firstObj);

        return NULL;

    }



    /* allocate the arrays */

    vcard_emul_alloc_arrays(&certs, &cert_len, &keys, cert_count);



    /* fill in the arrays */

    cert_count = 0;

    for (thisObj = firstObj; thisObj;

                             thisObj = PK11_GetNextGenericObject(thisObj)) {

        SECItem derCert;

        CERTCertificate *cert;

        SECStatus rv;



        rv = PK11_ReadRawAttribute(PK11_TypeGeneric, thisObj,

                                   CKA_VALUE, &derCert);

        if (rv != SECSuccess) {

            continue;

        }

        /* create floating temp cert. This gives us a cert structure even if

         * the token isn't logged in */

        cert = CERT_NewTempCertificate(CERT_GetDefaultCertDB(), &derCert,

                                       NULL, PR_FALSE, PR_TRUE);

        SECITEM_FreeItem(&derCert, PR_FALSE);

        if (cert == NULL) {

            continue;

        }



        certs[cert_count] = cert->derCert.data;

        cert_len[cert_count] = cert->derCert.len;

        keys[cert_count] = vcard_emul_make_key(slot, cert);

        cert_count++;

        CERT_DestroyCertificate(cert); /* key obj still has a reference */

    }



    /* now create the card */

    card = vcard_emul_make_card(vreader, certs, cert_len, keys, cert_count);

    g_free(certs);

    g_free(cert_len);

    g_free(keys);



    return card;

}
