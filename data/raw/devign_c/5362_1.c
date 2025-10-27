static int s390x_write_all_elf64_notes(const char *note_name,

                                       WriteCoreDumpFunction f,

                                       S390CPU *cpu, int id,

                                       void *opaque)

{

    Note note;

    const NoteFuncDesc *nf;

    int note_size;

    int ret = -1;



    for (nf = note_func; nf->note_contents_func; nf++) {

        memset(&note, 0, sizeof(note));

        note.hdr.n_namesz = cpu_to_be32(sizeof(note.name));

        note.hdr.n_descsz = cpu_to_be32(nf->contents_size);

        strncpy(note.name, note_name, sizeof(note.name));

        (*nf->note_contents_func)(&note, cpu);



        note_size = sizeof(note) - sizeof(note.contents) + nf->contents_size;

        ret = f(&note, note_size, opaque);



        if (ret < 0) {

            return -1;

        }



    }



    return 0;

}
