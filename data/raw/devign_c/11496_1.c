static void *handle_apdu_thread(void* arg)

{

    EmulatedState *card = arg;

    uint8_t recv_data[APDU_BUF_SIZE];

    int recv_len;

    VReaderStatus reader_status;

    EmulEvent *event;



    while (1) {

        qemu_mutex_lock(&card->handle_apdu_mutex);

        qemu_cond_wait(&card->handle_apdu_cond, &card->handle_apdu_mutex);

        qemu_mutex_unlock(&card->handle_apdu_mutex);

        if (card->quit_apdu_thread) {

            card->quit_apdu_thread = 0; /* debugging */

            break;

        }

        qemu_mutex_lock(&card->vreader_mutex);

        while (!QSIMPLEQ_EMPTY(&card->guest_apdu_list)) {

            event = QSIMPLEQ_FIRST(&card->guest_apdu_list);

            assert((unsigned long)event > 1000);

            QSIMPLEQ_REMOVE_HEAD(&card->guest_apdu_list, entry);

            if (event->p.data.type != EMUL_GUEST_APDU) {

                DPRINTF(card, 1, "unexpected message in handle_apdu_thread\n");

                g_free(event);

                continue;

            }

            if (card->reader == NULL) {

                DPRINTF(card, 1, "reader is NULL\n");

                g_free(event);

                continue;

            }

            recv_len = sizeof(recv_data);

            reader_status = vreader_xfr_bytes(card->reader,

                    event->p.data.data, event->p.data.len,

                    recv_data, &recv_len);

            DPRINTF(card, 2, "got back apdu of length %d\n", recv_len);

            if (reader_status == VREADER_OK) {

                emulated_push_response_apdu(card, recv_data, recv_len);

            } else {

                emulated_push_error(card, reader_status);

            }

            g_free(event);

        }

        qemu_mutex_unlock(&card->vreader_mutex);

    }

    qemu_mutex_lock(&card->apdu_thread_quit_mutex);

    qemu_cond_signal(&card->apdu_thread_quit_cond);

    qemu_mutex_unlock(&card->apdu_thread_quit_mutex);

    return NULL;

}
