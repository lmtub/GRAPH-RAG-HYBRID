vreader_list_delete(VReaderList *list)

{

    VReaderListEntry *current_entry;

    VReaderListEntry *next_entry = NULL;

    for (current_entry = vreader_list_get_first(list); current_entry;

         current_entry = next_entry) {

        next_entry = vreader_list_get_next(current_entry);

        vreader_list_entry_delete(current_entry);

    }

    list->head = NULL;

    list->tail = NULL;

    g_free(list);

}
