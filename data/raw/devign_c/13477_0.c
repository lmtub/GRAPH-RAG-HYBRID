static void print_tag(const char *str, unsigned int tag, int size)

{

    dprintf(NULL, "%s: tag=%c%c%c%c size=0x%x\n",

           str, tag & 0xff,

           (tag >> 8) & 0xff,

           (tag >> 16) & 0xff,

           (tag >> 24) & 0xff,

           size);

}
