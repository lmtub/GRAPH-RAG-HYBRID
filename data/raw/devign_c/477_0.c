static int img_open_password(BlockBackend *blk, const char *filename,

                             int flags, bool quiet)

{

    BlockDriverState *bs;

    char password[256];



    bs = blk_bs(blk);

    if (bdrv_is_encrypted(bs) && !(flags & BDRV_O_NO_IO)) {

        qprintf(quiet, "Disk image '%s' is encrypted.\n", filename);

        if (qemu_read_password(password, sizeof(password)) < 0) {

            error_report("No password given");

            return -1;

        }

        if (bdrv_set_key(bs, password) < 0) {

            error_report("invalid password");

            return -1;

        }

    }

    return 0;

}
