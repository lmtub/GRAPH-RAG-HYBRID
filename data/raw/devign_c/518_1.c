static int fdctrl_connect_drives(FDCtrl *fdctrl)

{

    unsigned int i;

    FDrive *drive;



    for (i = 0; i < MAX_FD; i++) {

        drive = &fdctrl->drives[i];

        drive->fdctrl = fdctrl;



        if (drive->bs) {

            if (bdrv_get_on_error(drive->bs, 0) != BLOCK_ERR_STOP_ENOSPC) {

                error_report("fdc doesn't support drive option werror");

                return -1;

            }

            if (bdrv_get_on_error(drive->bs, 1) != BLOCK_ERR_REPORT) {

                error_report("fdc doesn't support drive option rerror");

                return -1;

            }

        }



        fd_init(drive);

        fd_revalidate(drive);

        if (drive->bs) {

            bdrv_set_dev_ops(drive->bs, &fdctrl_block_ops, drive);

        }

    }

    return 0;

}
