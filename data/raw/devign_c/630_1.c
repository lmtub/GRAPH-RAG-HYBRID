void bdrv_get_backing_filename(BlockDriverState *bs,

                               char *filename, int filename_size)

{

    if (!bs->backing_hd) {

        pstrcpy(filename, filename_size, "");

    } else {

        pstrcpy(filename, filename_size, bs->backing_file);

    }

}
