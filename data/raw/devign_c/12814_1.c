static void read_sgi_header(ByteIOContext *f, SGIInfo *info)
{
    info->magic = (unsigned short) get_be16(f);
    info->rle = get_byte(f);
    info->bytes_per_channel = get_byte(f);
    info->dimension = (unsigned short)get_be16(f);
    info->xsize = (unsigned short) get_be16(f);
    info->ysize = (unsigned short) get_be16(f);
    info->zsize = (unsigned short) get_be16(f);
#ifdef DEBUG
    printf("sgi header fields:\n");
    printf("  magic: %d\n", info->magic);
    printf("    rle: %d\n", info->rle);
    printf("    bpc: %d\n", info->bytes_per_channel);
    printf("    dim: %d\n", info->dimension);
    printf("  xsize: %d\n", info->xsize);
    printf("  ysize: %d\n", info->ysize);
    printf("  zsize: %d\n", info->zsize);
#endif
    return;
}