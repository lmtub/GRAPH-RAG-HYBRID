static ram_addr_t find_ram_offset(ram_addr_t size)

{

    RAMBlock *block, *next_block;

    ram_addr_t offset = RAM_ADDR_MAX, mingap = RAM_ADDR_MAX;



    assert(size != 0); /* it would hand out same offset multiple times */



    if (QTAILQ_EMPTY(&ram_list.blocks))

        return 0;



    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        ram_addr_t end, next = RAM_ADDR_MAX;



        end = block->offset + block->length;



        QTAILQ_FOREACH(next_block, &ram_list.blocks, next) {

            if (next_block->offset >= end) {

                next = MIN(next, next_block->offset);

            }

        }

        if (next - end >= size && next - end < mingap) {

            offset = end;

            mingap = next - end;

        }

    }



    if (offset == RAM_ADDR_MAX) {

        fprintf(stderr, "Failed to find gap of requested size: %" PRIu64 "\n",

                (uint64_t)size);

        abort();

    }



    return offset;

}
