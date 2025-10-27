ssize_t cpu_get_note_size(int class, int machine, int nr_cpus)

{

    int name_size = 8; /* "CORE" or "QEMU" rounded */

    size_t elf_note_size = 0;

    int note_head_size;

    const NoteFuncDesc *nf;



    assert(class == ELFCLASS64);

    assert(machine == EM_S390);



    note_head_size = sizeof(Elf64_Nhdr);



    for (nf = note_func; nf->note_contents_func; nf++) {

        elf_note_size = elf_note_size + note_head_size + name_size +

                        nf->contents_size;

    }



    return (elf_note_size) * nr_cpus;

}
