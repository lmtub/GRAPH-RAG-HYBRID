static int parse_block_size_shift(BDRVSheepdogState *s, QemuOpts *opt)

{

    struct SheepdogInode *inode = &s->inode;

    uint64_t object_size;

    int obj_order;



    object_size = qemu_opt_get_size_del(opt, BLOCK_OPT_OBJECT_SIZE, 0);

    if (object_size) {

        if ((object_size - 1) & object_size) {    /* not a power of 2? */

            return -EINVAL;

        }

        obj_order = ffs(object_size) - 1;

        if (obj_order < 20 || obj_order > 31) {

            return -EINVAL;

        }

        inode->block_size_shift = (uint8_t)obj_order;

    }



    return 0;

}
