static hwaddr get_offset(hwaddr phys_addr,

                                     DumpState *s)

{

    RAMBlock *block;

    hwaddr offset = s->memory_offset;

    int64_t size_in_block, start;



    if (s->has_filter) {

        if (phys_addr < s->begin || phys_addr >= s->begin + s->length) {

            return -1;

        }

    }



    QTAILQ_FOREACH(block, &ram_list.blocks, next) {

        if (s->has_filter) {

            if (block->offset >= s->begin + s->length ||

                block->offset + block->length <= s->begin) {

                /* This block is out of the range */

                continue;

            }



            if (s->begin <= block->offset) {

                start = block->offset;

            } else {

                start = s->begin;

            }



            size_in_block = block->length - (start - block->offset);

            if (s->begin + s->length < block->offset + block->length) {

                size_in_block -= block->offset + block->length -

                                 (s->begin + s->length);

            }

        } else {

            start = block->offset;

            size_in_block = block->length;

        }



        if (phys_addr >= start && phys_addr < start + size_in_block) {

            return phys_addr - start + offset;

        }



        offset += size_in_block;

    }



    return -1;

}
