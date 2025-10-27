vcard_free(VCard *vcard)

{

    VCardApplet *current_applet = NULL;

    VCardApplet *next_applet = NULL;



    if (vcard == NULL) {

        return;

    }

    vcard->reference_count--;

    if (vcard->reference_count != 0) {

        return;

    }

    if (vcard->vcard_private_free) {

        (*vcard->vcard_private_free)(vcard->vcard_private);

        vcard->vcard_private_free = 0;

        vcard->vcard_private = 0;

    }

    for (current_applet = vcard->applet_list; current_applet;

                                        current_applet = next_applet) {

        next_applet = current_applet->next;

        vcard_delete_applet(current_applet);

    }

    vcard_buffer_response_delete(vcard->vcard_buffer_response);

    g_free(vcard);

}
