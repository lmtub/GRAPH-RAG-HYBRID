static int qcow2_change_backing_file(BlockDriverState *bs,

    const char *backing_file, const char *backing_fmt)

{

    /* Backing file format doesn't make sense without a backing file */

    if (backing_fmt && !backing_file) {

        return -EINVAL;

    }



    pstrcpy(bs->backing_file, sizeof(bs->backing_file), backing_file ?: "");

    pstrcpy(bs->backing_format, sizeof(bs->backing_format), backing_fmt ?: "");



    return qcow2_update_header(bs);

}
