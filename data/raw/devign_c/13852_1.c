static int openfile(char *name, int flags, int growable, QDict *opts)

{

    Error *local_err = NULL;



    if (qemuio_bs) {

        fprintf(stderr, "file open already, try 'help close'\n");


        return 1;

    }



    if (growable) {

        if (bdrv_open(&qemuio_bs, name, NULL, opts, flags | BDRV_O_PROTOCOL,

                      NULL, &local_err))

        {

            fprintf(stderr, "%s: can't open device %s: %s\n", progname, name,

                    error_get_pretty(local_err));

            error_free(local_err);

            return 1;

        }

    } else {

        qemuio_bs = bdrv_new("hda", &error_abort);



        if (bdrv_open(&qemuio_bs, name, NULL, opts, flags, NULL, &local_err)

            < 0)

        {

            fprintf(stderr, "%s: can't open device %s: %s\n", progname, name,

                    error_get_pretty(local_err));

            error_free(local_err);

            bdrv_unref(qemuio_bs);

            qemuio_bs = NULL;

            return 1;

        }

    }



    return 0;

}