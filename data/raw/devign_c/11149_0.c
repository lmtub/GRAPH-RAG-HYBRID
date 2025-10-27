void bdrv_info(void)

{

    BlockDriverState *bs;



    for (bs = bdrv_first; bs != NULL; bs = bs->next) {

        term_printf("%s:", bs->device_name);

        term_printf(" type=");

        switch(bs->type) {

        case BDRV_TYPE_HD:

            term_printf("hd");

            break;

        case BDRV_TYPE_CDROM:

            term_printf("cdrom");

            break;

        case BDRV_TYPE_FLOPPY:

            term_printf("floppy");

            break;

        }

        term_printf(" removable=%d", bs->removable);

        if (bs->removable) {

            term_printf(" locked=%d", bs->locked);

        }

        if (bs->drv) {

            term_printf(" file=");

	    term_print_filename(bs->filename);

            if (bs->backing_file[0] != '\0') {

                term_printf(" backing_file=");

		term_print_filename(bs->backing_file);

	    }

            term_printf(" ro=%d", bs->read_only);

            term_printf(" drv=%s", bs->drv->format_name);

            if (bs->encrypted)

                term_printf(" encrypted");

        } else {

            term_printf(" [not inserted]");

        }

        term_printf("\n");

    }

}
