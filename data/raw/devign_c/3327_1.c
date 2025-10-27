static int qcow2_change_backing_file(BlockDriverState *bs,

    const char *backing_file, const char *backing_fmt)

{

    return qcow2_update_ext_header(bs, backing_file, backing_fmt);

}
