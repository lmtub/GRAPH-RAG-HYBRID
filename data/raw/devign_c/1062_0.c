do_socket_read(GIOChannel *source,

               GIOCondition condition,

               gpointer data)

{

    int rv;

    int dwSendLength;

    int dwRecvLength;

    uint8_t pbRecvBuffer[APDUBufSize];

    static uint8_t pbSendBuffer[APDUBufSize];

    VReaderStatus reader_status;

    VReader *reader = NULL;

    static VSCMsgHeader mhHeader;

    VSCMsgError *error_msg;

    GError *err = NULL;



    static gchar *buf;

    static gsize br, to_read;

    static int state = STATE_HEADER;



    if (state == STATE_HEADER && to_read == 0) {

        buf = (gchar *)&mhHeader;

        to_read = sizeof(mhHeader);

    }



    if (to_read > 0) {

        g_io_channel_read_chars(source, (gchar *)buf, to_read, &br, &err);

        if (err != NULL) {

            g_error("error while reading: %s", err->message);

        }

        buf += br;

        to_read -= br;

        if (to_read != 0) {

            return TRUE;

        }

    }



    if (state == STATE_HEADER) {

        mhHeader.type = ntohl(mhHeader.type);

        mhHeader.reader_id = ntohl(mhHeader.reader_id);

        mhHeader.length = ntohl(mhHeader.length);

        if (verbose) {

            printf("Header: type=%d, reader_id=%u length=%d (0x%x)\n",

                   mhHeader.type, mhHeader.reader_id, mhHeader.length,

                   mhHeader.length);

        }

        switch (mhHeader.type) {

        case VSC_APDU:

        case VSC_Flush:

        case VSC_Error:

        case VSC_Init:

            buf = (gchar *)pbSendBuffer;

            to_read = mhHeader.length;

            state = STATE_MESSAGE;

            return TRUE;

        default:

            fprintf(stderr, "Unexpected message of type 0x%X\n", mhHeader.type);

            return FALSE;

        }

    }



    if (state == STATE_MESSAGE) {

        switch (mhHeader.type) {

        case VSC_APDU:

            if (verbose) {

                printf(" recv APDU: ");

                print_byte_array(pbSendBuffer, mhHeader.length);

            }

            /* Transmit received APDU */

            dwSendLength = mhHeader.length;

            dwRecvLength = sizeof(pbRecvBuffer);

            reader = vreader_get_reader_by_id(mhHeader.reader_id);

            reader_status = vreader_xfr_bytes(reader,

                                              pbSendBuffer, dwSendLength,

                                              pbRecvBuffer, &dwRecvLength);

            if (reader_status == VREADER_OK) {

                mhHeader.length = dwRecvLength;

                if (verbose) {

                    printf(" send response: ");

                    print_byte_array(pbRecvBuffer, mhHeader.length);

                }

                send_msg(VSC_APDU, mhHeader.reader_id,

                         pbRecvBuffer, dwRecvLength);

            } else {

                rv = reader_status; /* warning: not meaningful */

                send_msg(VSC_Error, mhHeader.reader_id, &rv, sizeof(uint32_t));

            }

            vreader_free(reader);

            reader = NULL; /* we've freed it, don't use it by accident

                              again */

            break;

        case VSC_Flush:

            /* TODO: actually flush */

            send_msg(VSC_FlushComplete, mhHeader.reader_id, NULL, 0);

            break;

        case VSC_Error:

            error_msg = (VSCMsgError *) pbSendBuffer;

            if (error_msg->code == VSC_SUCCESS) {

                qemu_mutex_lock(&pending_reader_lock);

                if (pending_reader) {

                    vreader_set_id(pending_reader, mhHeader.reader_id);

                    vreader_free(pending_reader);

                    pending_reader = NULL;

                    qemu_cond_signal(&pending_reader_condition);

                }

                qemu_mutex_unlock(&pending_reader_lock);

                break;

            }

            printf("warning: qemu refused to add reader\n");

            if (error_msg->code == VSC_CANNOT_ADD_MORE_READERS) {

                /* clear pending reader, qemu can't handle any more */

                qemu_mutex_lock(&pending_reader_lock);

                if (pending_reader) {

                    pending_reader = NULL;

                    /* make sure the event loop doesn't hang */

                    qemu_cond_signal(&pending_reader_condition);

                }

                qemu_mutex_unlock(&pending_reader_lock);

            }

            break;

        case VSC_Init:

            if (on_host_init(&mhHeader, (VSCMsgInit *)pbSendBuffer) < 0) {

                return FALSE;

            }

            break;

        default:

            g_assert_not_reached();

            return FALSE;

        }



        state = STATE_HEADER;

    }





    return TRUE;

}
