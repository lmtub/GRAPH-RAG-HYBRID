static void fill_note(struct memelfnote *note, const char *name, int type,

                      unsigned int sz, void *data)

{

    unsigned int namesz;



    namesz = strlen(name) + 1;

    note->name = name;

    note->namesz = namesz;

    note->namesz_rounded = roundup(namesz, sizeof (int32_t));

    note->type = type;

    note->datasz = roundup(sz, sizeof (int32_t));;

    note->data = data;



    /*

     * We calculate rounded up note size here as specified by

     * ELF document.

     */

    note->notesz = sizeof (struct elf_note) +

        note->namesz_rounded + note->datasz;

}
