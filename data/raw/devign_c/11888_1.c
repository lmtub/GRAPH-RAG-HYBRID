vcard_emul_replay_insertion_events(void)
{
    VReaderListEntry *current_entry;
    VReaderListEntry *next_entry = NULL;
    VReaderList *list = vreader_get_reader_list();
    for (current_entry = vreader_list_get_first(list); current_entry;
            current_entry = next_entry) {
        VReader *vreader = vreader_list_get_reader(current_entry);
        next_entry = vreader_list_get_next(current_entry);
        vreader_queue_card_event(vreader);
    }
}