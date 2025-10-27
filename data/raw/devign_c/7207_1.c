static int dump_iterate(DumpState *s)

{

    RAMBlock *block;

    int64_t size;

    int ret;



    while (1) {

        block = s->block;



        size = block->length;

        if (s->has_filter) {

            size -= s->start;

            if (s->begin + s->length < block->offset + block->length) {

                size -= block->offset + block->length - (s->begin + s->length);

            }

        }

        ret = write_memory(s, block, s->start, size);

        if (ret == -1) {

            return ret;

        }



        ret = get_next_block(s, block);

        if (ret == 1) {

            dump_completed(s);

            return 0;

        }

    }

}
