static off_t read_off(int fd, int64_t offset)

{

	uint64_t buffer;

	if (pread(fd, &buffer, 8, offset) < 8)

		return 0;

	return be64_to_cpu(buffer);

}
