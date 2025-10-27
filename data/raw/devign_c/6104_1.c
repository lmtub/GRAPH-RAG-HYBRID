static int emulated_initfn(CCIDCardState *base)

{

    EmulatedState *card = DO_UPCAST(EmulatedState, base, base);

    VCardEmulError ret;

    const EnumTable *ptable;



    QSIMPLEQ_INIT(&card->event_list);

    QSIMPLEQ_INIT(&card->guest_apdu_list);

    qemu_mutex_init(&card->event_list_mutex);

    qemu_mutex_init(&card->vreader_mutex);

    qemu_mutex_init(&card->handle_apdu_mutex);

    qemu_cond_init(&card->handle_apdu_cond);

    card->reader = NULL;

    card->quit_apdu_thread = 0;

    if (init_pipe_signaling(card) < 0) {

        return -1;

    }

    card->backend = parse_enumeration(card->backend_str, backend_enum_table, 0);

    if (card->backend == 0) {

        printf("unknown backend, must be one of:\n");

        for (ptable = backend_enum_table; ptable->name != NULL; ++ptable) {

            printf("%s\n", ptable->name);

        }

        return -1;

    }



    /* TODO: a passthru backened that works on local machine. third card type?*/

    if (card->backend == BACKEND_CERTIFICATES) {

        if (card->cert1 != NULL && card->cert2 != NULL && card->cert3 != NULL) {

            ret = emulated_initialize_vcard_from_certificates(card);

        } else {

            printf("%s: you must provide all three certs for"

                   " certificates backend\n", EMULATED_DEV_NAME);

            return -1;

        }

    } else {

        if (card->backend != BACKEND_NSS_EMULATED) {

            printf("%s: bad backend specified. The options are:\n%s (default),"

                " %s.\n", EMULATED_DEV_NAME, BACKEND_NSS_EMULATED_NAME,

                BACKEND_CERTIFICATES_NAME);

            return -1;

        }

        if (card->cert1 != NULL || card->cert2 != NULL || card->cert3 != NULL) {

            printf("%s: unexpected cert parameters to nss emulated backend\n",

                   EMULATED_DEV_NAME);

            return -1;

        }

        /* default to mirroring the local hardware readers */

        ret = wrap_vcard_emul_init(NULL);

    }

    if (ret != VCARD_EMUL_OK) {

        printf("%s: failed to initialize vcard\n", EMULATED_DEV_NAME);

        return -1;

    }

    qemu_thread_create(&card->event_thread_id, event_thread, card,

                       QEMU_THREAD_JOINABLE);

    qemu_thread_create(&card->apdu_thread_id, handle_apdu_thread, card,

                       QEMU_THREAD_JOINABLE);

    return 0;

}
