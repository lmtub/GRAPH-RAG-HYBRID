static void do_eject(int argc, const char **argv)

{

    BlockDriverState *bs;

    const char **parg;

    int force;



    parg = argv + 1;

    if (!*parg) {

    fail:

        help_cmd(argv[0]);

        return;

    }

    force = 0;

    if (!strcmp(*parg, "-f")) {

        force = 1;

        parg++;

    }

    if (!*parg)

        goto fail;

    bs = bdrv_find(*parg);

    if (!bs) {

        term_printf("device not found\n");

        return;

    }

    eject_device(bs, force);

}
