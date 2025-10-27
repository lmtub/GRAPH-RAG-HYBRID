CharDriverState *qemu_chr_find(const char *name)

{

    CharDriverState *chr;



    TAILQ_FOREACH(chr, &chardevs, next) {

        if (strcmp(chr->label, name) != 0)

            continue;

        return chr;

    }

    return NULL;

}
