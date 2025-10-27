static void get_offset_range(hwaddr phys_addr,

                             ram_addr_t mapping_length,

                             DumpState *s,

                             hwaddr *p_offset,

                             hwaddr *p_filesz)

{

    RAMBlock *block;

    hwaddr offset = s->memory_offset;

    int64_t size_in_block, start;



    /* When the memory is not stored into vmcore, offset will be -1 */

    *p_offset = -1;

    *p_filesz = 0;



    if (s->has_filter) {

        if (phys_addr < s->begin || phys_addr >= s->begin + s->length) {

            return;

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

            *p_offset = phys_addr - start + offset;



            /* The offset range mapped from the vmcore file must not spill over

             * the RAMBlock, clamp it. The rest of the mapping will be

             * zero-filled in memory at load time; see

             * <http://refspecs.linuxbase.org/elf/gabi4+/ch5.pheader.html>.

             */

            *p_filesz = phys_addr + mapping_length <= start + size_in_block ?

                        mapping_length :

                        size_in_block - (phys_addr - start);

            return;

        }



        offset += size_in_block;

    }

}
