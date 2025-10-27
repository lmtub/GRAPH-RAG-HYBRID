static off_t read_uint32(BlockDriverState *bs, int64_t offset)

{

	uint32_t buffer;

	if (bdrv_pread(bs->file, offset, &buffer, 4) < 4)

		return 0;

	return be32_to_cpu(buffer);

}
