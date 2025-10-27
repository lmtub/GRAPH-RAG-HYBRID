static int vmdk_open_desc_file(BlockDriverState *bs, int flags,

                               int64_t desc_offset)

{

    int ret;

    char buf[2048];

    char ct[128];

    BDRVVmdkState *s = bs->opaque;



    ret = bdrv_pread(bs->file, desc_offset, buf, sizeof(buf));

    if (ret < 0) {

        return ret;

    }

    buf[2047] = '\0';

    if (vmdk_parse_description(buf, "createType", ct, sizeof(ct))) {

        return -EINVAL;

    }

    if (strcmp(ct, "monolithicFlat") &&

        strcmp(ct, "twoGbMaxExtentSparse") &&

        strcmp(ct, "twoGbMaxExtentFlat")) {

        fprintf(stderr,

                "VMDK: Not supported image type \"%s\""".\n", ct);

        return -ENOTSUP;

    }

    s->desc_offset = 0;

    ret = vmdk_parse_extents(buf, bs, bs->file->filename);

    if (ret) {

        return ret;

    }



    /* try to open parent images, if exist */

    if (vmdk_parent_open(bs)) {

        g_free(s->extents);

        return -EINVAL;

    }

    s->parent_cid = vmdk_read_cid(bs, 1);

    return 0;

}
