static int raw_truncate(BlockDriverState *bs, int64_t offset)

{

    BDRVRawState *s = bs->opaque;

    LONG low, high;



    low = offset;

    high = offset >> 32;

    if (!SetFilePointer(s->hfile, low, &high, FILE_BEGIN))

	return -EIO;

    if (!SetEndOfFile(s->hfile))

        return -EIO;

    return 0;

}
