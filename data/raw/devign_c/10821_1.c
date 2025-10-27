static int get_next_block(DumpState *s, RAMBlock *block)

{

    while (1) {

        block = QTAILQ_NEXT(block, next);

        if (!block) {

            /* no more block */

            return 1;

        }



        s->start = 0;

        s->block = block;

        if (s->has_filter) {

            if (block->offset >= s->begin + s->length ||

                block->offset + block->length <= s->begin) {

                /* This block is out of the range */

                continue;

            }



            if (s->begin > block->offset) {

                s->start = s->begin - block->offset;

            }

        }



        return 0;

    }

}
