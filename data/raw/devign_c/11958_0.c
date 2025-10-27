static int vmdk_parent_open(BlockDriverState *bs)

{

    char *p_name;

    char desc[DESC_SIZE + 1];

    BDRVVmdkState *s = bs->opaque;



    desc[DESC_SIZE] = '\0';

    if (bdrv_pread(bs->file, s->desc_offset, desc, DESC_SIZE) != DESC_SIZE) {

        return -1;

    }



    if ((p_name = strstr(desc,"parentFileNameHint")) != NULL) {

        char *end_name;



        p_name += sizeof("parentFileNameHint") + 1;

        if ((end_name = strchr(p_name,'\"')) == NULL)

            return -1;

        if ((end_name - p_name) > sizeof (bs->backing_file) - 1)

            return -1;



        pstrcpy(bs->backing_file, end_name - p_name + 1, p_name);

    }



    return 0;

}
