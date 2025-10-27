static int write_note(struct memelfnote *men, int fd)

{

    struct elf_note en;



    en.n_namesz = men->namesz;

    en.n_type = men->type;

    en.n_descsz = men->datasz;



    bswap_note(&en);



    if (dump_write(fd, &en, sizeof(en)) != 0)

        return (-1);

    if (dump_write(fd, men->name, men->namesz_rounded) != 0)

        return (-1);

    if (dump_write(fd, men->data, men->datasz) != 0)

        return (-1);



    return (0);

}
