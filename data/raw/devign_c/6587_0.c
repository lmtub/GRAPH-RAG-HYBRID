static off_t read_uint32(int fd, int64_t offset)

{

	uint32_t buffer;

	if (pread(fd, &buffer, 4, offset) < 4)

		return 0;

	return be32_to_cpu(buffer);

}
