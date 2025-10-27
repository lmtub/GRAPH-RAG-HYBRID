static void open_help(void)

{

    printf(

"\n"

" opens a new file in the requested mode\n"

"\n"

" Example:\n"

" 'open -Cn /tmp/data' - creates/opens data file read-write and uncached\n"

"\n"

" Opens a file for subsequent use by all of the other qemu-io commands.\n"

" -r, -- open file read-only\n"

" -s, -- use snapshot file\n"

" -n, -- disable host cache\n"

" -o, -- options to be given to the block driver"

"\n");

}
