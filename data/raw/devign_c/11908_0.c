const char *bdrv_get_encrypted_filename(BlockDriverState *bs)

{

    if (bs->backing_hd && bs->backing_hd->encrypted)

        return bs->backing_file;

    else if (bs->encrypted)

        return bs->filename;

    else

        return NULL;

}
