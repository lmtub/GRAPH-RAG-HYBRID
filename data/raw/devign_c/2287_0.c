event_thread(void *arg)

{

    unsigned char atr[MAX_ATR_LEN];

    int atr_len = MAX_ATR_LEN;

    VEvent *event = NULL;

    unsigned int reader_id;





    while (1) {

        const char *reader_name;



        event = vevent_wait_next_vevent();

        if (event == NULL) {

            break;

        }

        reader_id = vreader_get_id(event->reader);

        if (reader_id == VSCARD_UNDEFINED_READER_ID &&

            event->type != VEVENT_READER_INSERT) {

            /* ignore events from readers qemu has rejected */

            /* if qemu is still deciding on this reader, wait to see if need to

             * forward this event */

            qemu_mutex_lock(&pending_reader_lock);

            if (!pending_reader || (pending_reader != event->reader)) {

                /* wasn't for a pending reader, this reader has already been

                 * rejected by qemu */

                qemu_mutex_unlock(&pending_reader_lock);

                vevent_delete(event);

                continue;

            }

            /* this reader hasn't been told its status from qemu yet, wait for

             * that status */

            while (pending_reader != NULL) {

                qemu_cond_wait(&pending_reader_condition, &pending_reader_lock);

            }

            qemu_mutex_unlock(&pending_reader_lock);

            /* now recheck the id */

            reader_id = vreader_get_id(event->reader);

            if (reader_id == VSCARD_UNDEFINED_READER_ID) {

                /* this reader was rejected */

                vevent_delete(event);

                continue;

            }

            /* reader was accepted, now forward the event */

        }

        switch (event->type) {

        case VEVENT_READER_INSERT:

            /* tell qemu to insert a new CCID reader */

            /* wait until qemu has responded to our first reader insert

             * before we send a second. That way we won't confuse the responses

             * */

            qemu_mutex_lock(&pending_reader_lock);

            while (pending_reader != NULL) {

                qemu_cond_wait(&pending_reader_condition, &pending_reader_lock);

            }

            pending_reader = vreader_reference(event->reader);

            qemu_mutex_unlock(&pending_reader_lock);

            reader_name = vreader_get_name(event->reader);

            if (verbose > 10) {

                printf(" READER INSERT: %s\n", reader_name);

            }

            send_msg(VSC_ReaderAdd,

                reader_id, /* currerntly VSCARD_UNDEFINED_READER_ID */

                NULL, 0 /* TODO reader_name, strlen(reader_name) */);

            break;

        case VEVENT_READER_REMOVE:

            /* future, tell qemu that an old CCID reader has been removed */

            if (verbose > 10) {

                printf(" READER REMOVE: %u\n", reader_id);

            }

            send_msg(VSC_ReaderRemove, reader_id, NULL, 0);

            break;

        case VEVENT_CARD_INSERT:

            /* get the ATR (intended as a response to a power on from the

             * reader */

            atr_len = MAX_ATR_LEN;

            vreader_power_on(event->reader, atr, &atr_len);

            /* ATR call functions as a Card Insert event */

            if (verbose > 10) {

                printf(" CARD INSERT %u: ", reader_id);

                print_byte_array(atr, atr_len);

            }

            send_msg(VSC_ATR, reader_id, atr, atr_len);

            break;

        case VEVENT_CARD_REMOVE:

            /* Card removed */

            if (verbose > 10) {

                printf(" CARD REMOVE %u:\n", reader_id);

            }

            send_msg(VSC_CardRemove, reader_id, NULL, 0);

            break;

        default:

            break;

        }

        vevent_delete(event);

    }

    return NULL;

}
