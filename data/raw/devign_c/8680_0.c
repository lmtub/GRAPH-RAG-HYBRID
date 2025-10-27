static int64_t raw_getlength(BlockDriverState *bs)

{

    int64_t len;

    BDRVRawState *s = bs->opaque;



    /* Update size. It should not change unless the file was externally

     * modified. */

    len = bdrv_getlength(bs->file->bs);

    if (len < 0) {

        return len;

    }



    if (len < s->offset) {

        s->size = 0;

    } else {

        if (s->has_size) {

            /* Try to honour the size */

            s->size = MIN(s->size, len - s->offset);

        } else {

            s->size = len - s->offset;

        }

    }



    return s->size;

}
