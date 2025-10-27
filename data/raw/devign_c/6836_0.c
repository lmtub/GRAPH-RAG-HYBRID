static int do_compress_ram_page(CompressParam *param)

{

    int bytes_sent, blen;

    uint8_t *p;

    RAMBlock *block = param->block;

    ram_addr_t offset = param->offset;



    p = block->host + (offset & TARGET_PAGE_MASK);



    bytes_sent = save_page_header(param->file, block, offset |

                                  RAM_SAVE_FLAG_COMPRESS_PAGE);

    blen = qemu_put_compression_data(param->file, p, TARGET_PAGE_SIZE,

                                     migrate_compress_level());

    bytes_sent += blen;



    return bytes_sent;

}
