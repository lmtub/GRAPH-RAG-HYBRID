static size_t save_page_header(RAMState *rs, RAMBlock *block, ram_addr_t offset)

{

    size_t size, len;



    if (block == rs->last_sent_block) {

        offset |= RAM_SAVE_FLAG_CONTINUE;

    }

    qemu_put_be64(rs->f, offset);

    size = 8;



    if (!(offset & RAM_SAVE_FLAG_CONTINUE)) {

        len = strlen(block->idstr);

        qemu_put_byte(rs->f, len);

        qemu_put_buffer(rs->f, (uint8_t *)block->idstr, len);

        size += 1 + len;

        rs->last_sent_block = block;

    }

    return size;

}
