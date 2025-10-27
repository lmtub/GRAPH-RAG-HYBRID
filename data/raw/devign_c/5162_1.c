static void vmdk_close(BlockDriverState *bs)

{

    BDRVVmdkState *s = bs->opaque;



    qemu_free(s->l1_table);

    qemu_free(s->l2_cache);

    bdrv_delete(s->hd);

    // try to close parent image, if exist

    vmdk_parent_close(s->hd);

}
