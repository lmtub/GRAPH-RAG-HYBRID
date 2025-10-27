static void do_change(int argc, const char **argv)

{

    BlockDriverState *bs;



    if (argc != 3) {

        help_cmd(argv[0]);

        return;

    }

    bs = bdrv_find(argv[1]);

    if (!bs) {

        term_printf("device not found\n");

        return;

    }

    if (eject_device(bs, 0) < 0)

        return;

    bdrv_open(bs, argv[2], 0);

}
