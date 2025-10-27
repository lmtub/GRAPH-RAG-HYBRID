static int img_snapshot(int argc, char **argv)

{

    BlockDriverState *bs;

    QEMUSnapshotInfo sn;

    char *filename, *snapshot_name = NULL;

    int c, ret = 0, bdrv_oflags;

    int action = 0;

    qemu_timeval tv;

    bool quiet = false;



    bdrv_oflags = BDRV_O_FLAGS | BDRV_O_RDWR;

    /* Parse commandline parameters */

    for(;;) {

        c = getopt(argc, argv, "la:c:d:hq");

        if (c == -1) {

            break;

        }

        switch(c) {

        case '?':

        case 'h':

            help();

            return 0;

        case 'l':

            if (action) {

                help();

                return 0;

            }

            action = SNAPSHOT_LIST;

            bdrv_oflags &= ~BDRV_O_RDWR; /* no need for RW */

            break;

        case 'a':

            if (action) {

                help();

                return 0;

            }

            action = SNAPSHOT_APPLY;

            snapshot_name = optarg;

            break;

        case 'c':

            if (action) {

                help();

                return 0;

            }

            action = SNAPSHOT_CREATE;

            snapshot_name = optarg;

            break;

        case 'd':

            if (action) {

                help();

                return 0;

            }

            action = SNAPSHOT_DELETE;

            snapshot_name = optarg;

            break;

        case 'q':

            quiet = true;

            break;

        }

    }



    if (optind != argc - 1) {

        help();

    }

    filename = argv[optind++];



    /* Open the image */

    bs = bdrv_new_open(filename, NULL, bdrv_oflags, true, quiet);

    if (!bs) {

        return 1;

    }



    /* Perform the requested action */

    switch(action) {

    case SNAPSHOT_LIST:

        dump_snapshots(bs);

        break;



    case SNAPSHOT_CREATE:

        memset(&sn, 0, sizeof(sn));

        pstrcpy(sn.name, sizeof(sn.name), snapshot_name);



        qemu_gettimeofday(&tv);

        sn.date_sec = tv.tv_sec;

        sn.date_nsec = tv.tv_usec * 1000;



        ret = bdrv_snapshot_create(bs, &sn);

        if (ret) {

            error_report("Could not create snapshot '%s': %d (%s)",

                snapshot_name, ret, strerror(-ret));

        }

        break;



    case SNAPSHOT_APPLY:

        ret = bdrv_snapshot_goto(bs, snapshot_name);

        if (ret) {

            error_report("Could not apply snapshot '%s': %d (%s)",

                snapshot_name, ret, strerror(-ret));

        }

        break;



    case SNAPSHOT_DELETE:

        ret = bdrv_snapshot_delete(bs, snapshot_name);

        if (ret) {

            error_report("Could not delete snapshot '%s': %d (%s)",

                snapshot_name, ret, strerror(-ret));

        }

        break;

    }



    /* Cleanup */

    bdrv_unref(bs);

    if (ret) {

        return 1;

    }

    return 0;

}
