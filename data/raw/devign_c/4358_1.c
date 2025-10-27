static off_t read_off(BlockDriverState *bs, int64_t offset)

{

	uint64_t buffer;

	if (bdrv_pread(bs->file, offset, &buffer, 8) < 8)

		return 0;

	return be64_to_cpu(buffer);

}
