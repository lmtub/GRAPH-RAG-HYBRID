static void multiwrite_help(void)

{

    printf(

"\n"

" writes a range of bytes from the given offset source from multiple buffers,\n"

" in a batch of requests that may be merged by qemu\n"

"\n"

" Example:\n"

" 'multiwrite 512 1k 1k ; 4k 1k' \n"

"  writes 2 kB at 512 bytes and 1 kB at 4 kB into the open file\n"

"\n"

" Writes into a segment of the currently open file, using a buffer\n"

" filled with a set pattern (0xcdcdcdcd). The pattern byte is increased\n"

" by one for each request contained in the multiwrite command.\n"

" -P, -- use different pattern to fill file\n"

" -C, -- report statistics in a machine parsable format\n"

" -q, -- quiet mode, do not show I/O statistics\n"

"\n");

}
