do_command(GIOChannel *source,

           GIOCondition condition,

           gpointer data)

{

    char *string;

    VCardEmulError error;

    static unsigned int default_reader_id;

    unsigned int reader_id;

    VReader *reader = NULL;

    GError *err = NULL;



    g_assert(condition & G_IO_IN);



    reader_id = default_reader_id;

    g_io_channel_read_line(source, &string, NULL, NULL, &err);

    if (err != NULL) {

        g_error("Error while reading command: %s", err->message);

    }



    if (string != NULL) {

        if (strncmp(string, "exit", 4) == 0) {

            /* remove all the readers */

            VReaderList *list = vreader_get_reader_list();

            VReaderListEntry *reader_entry;

            printf("Active Readers:\n");

            for (reader_entry = vreader_list_get_first(list); reader_entry;

                 reader_entry = vreader_list_get_next(reader_entry)) {

                VReader *reader = vreader_list_get_reader(reader_entry);

                vreader_id_t reader_id;

                reader_id = vreader_get_id(reader);

                if (reader_id == -1) {

                    continue;

                }

                /* be nice and signal card removal first (qemu probably should

                 * do this itself) */

                if (vreader_card_is_present(reader) == VREADER_OK) {

                    send_msg(VSC_CardRemove, reader_id, NULL, 0);

                }

                send_msg(VSC_ReaderRemove, reader_id, NULL, 0);

            }

            exit(0);

        } else if (strncmp(string, "insert", 6) == 0) {

            if (string[6] == ' ') {

                reader_id = get_id_from_string(&string[7], reader_id);

            }

            reader = vreader_get_reader_by_id(reader_id);

            if (reader != NULL) {

                error = vcard_emul_force_card_insert(reader);

                printf("insert %s, returned %d\n",

                       vreader_get_name(reader), error);

            } else {

                printf("no reader by id %u found\n", reader_id);

            }

        } else if (strncmp(string, "remove", 6) == 0) {

            if (string[6] == ' ') {

                reader_id = get_id_from_string(&string[7], reader_id);

            }

            reader = vreader_get_reader_by_id(reader_id);

            if (reader != NULL) {

                error = vcard_emul_force_card_remove(reader);

                printf("remove %s, returned %d\n",

                       vreader_get_name(reader), error);

            } else {

                printf("no reader by id %u found\n", reader_id);

            }

        } else if (strncmp(string, "select", 6) == 0) {

            if (string[6] == ' ') {

                reader_id = get_id_from_string(&string[7],

                                               VSCARD_UNDEFINED_READER_ID);

            }

            if (reader_id != VSCARD_UNDEFINED_READER_ID) {

                reader = vreader_get_reader_by_id(reader_id);

            }

            if (reader) {

                printf("Selecting reader %u, %s\n", reader_id,

                        vreader_get_name(reader));

                default_reader_id = reader_id;

            } else {

                printf("Reader with id %u not found\n", reader_id);

            }

        } else if (strncmp(string, "debug", 5) == 0) {

            if (string[5] == ' ') {

                verbose = get_id_from_string(&string[6], 0);

            }

            printf("debug level = %d\n", verbose);

        } else if (strncmp(string, "list", 4) == 0) {

            VReaderList *list = vreader_get_reader_list();

            VReaderListEntry *reader_entry;

            printf("Active Readers:\n");

            for (reader_entry = vreader_list_get_first(list); reader_entry;

                 reader_entry = vreader_list_get_next(reader_entry)) {

                VReader *reader = vreader_list_get_reader(reader_entry);

                vreader_id_t reader_id;

                reader_id = vreader_get_id(reader);

                if (reader_id == -1) {

                    continue;

                }

                printf("%3u %s %s\n", reader_id,

                       vreader_card_is_present(reader) == VREADER_OK ?

                       "CARD_PRESENT" : "            ",

                       vreader_get_name(reader));

            }

            printf("Inactive Readers:\n");

            for (reader_entry = vreader_list_get_first(list); reader_entry;

                 reader_entry = vreader_list_get_next(reader_entry)) {

                VReader *reader = vreader_list_get_reader(reader_entry);

                vreader_id_t reader_id;

                reader_id = vreader_get_id(reader);

                if (reader_id != -1) {

                    continue;

                }



                printf("INA %s %s\n",

                       vreader_card_is_present(reader) == VREADER_OK ?

                       "CARD_PRESENT" : "            ",

                       vreader_get_name(reader));

            }


        } else if (*string != 0) {

            printf("valid commands:\n");

            printf("insert [reader_id]\n");

            printf("remove [reader_id]\n");

            printf("select reader_id\n");

            printf("list\n");

            printf("debug [level]\n");

            printf("exit\n");

        }

    }

    vreader_free(reader);

    printf("> ");

    fflush(stdout);



    return TRUE;

}