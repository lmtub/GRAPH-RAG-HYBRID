static uint32_t vmdk_read_cid(BlockDriverState *bs, int parent)

{

    char desc[DESC_SIZE];

    uint32_t cid;

    const char *p_name, *cid_str;

    size_t cid_str_size;

    BDRVVmdkState *s = bs->opaque;



    if (bdrv_pread(bs->file, s->desc_offset, desc, DESC_SIZE) != DESC_SIZE) {

        return 0;

    }



    if (parent) {

        cid_str = "parentCID";

        cid_str_size = sizeof("parentCID");

    } else {

        cid_str = "CID";

        cid_str_size = sizeof("CID");

    }



    p_name = strstr(desc, cid_str);

    if (p_name != NULL) {

        p_name += cid_str_size;

        sscanf(p_name, "%x", &cid);

    }



    return cid;

}
