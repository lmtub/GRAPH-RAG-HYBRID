static int id_wellformed(const char *id)

{

    int i;



    if (!qemu_isalpha(id[0])) {

        return 0;

    }

    for (i = 1; id[i]; i++) {

        if (!qemu_isalnum(id[i]) && !strchr("-._", id[i])) {

            return 0;

        }

    }

    return 1;

}
