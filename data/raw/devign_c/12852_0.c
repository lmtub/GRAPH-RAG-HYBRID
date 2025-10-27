static int vmdk_open(BlockDriverState *bs, int flags)

{

    int ret;

    BDRVVmdkState *s = bs->opaque;



    if (vmdk_open_sparse(bs, bs->file, flags) == 0) {

        s->desc_offset = 0x200;

    } else {

        ret = vmdk_open_desc_file(bs, flags, 0);

        if (ret) {

            goto fail;

        }

    }

    /* try to open parent images, if exist */

    ret = vmdk_parent_open(bs);

    if (ret) {

        goto fail;

    }

    s->parent_cid = vmdk_read_cid(bs, 1);

    qemu_co_mutex_init(&s->lock);

    return ret;



fail:

    vmdk_free_extents(bs);

    return ret;

}
