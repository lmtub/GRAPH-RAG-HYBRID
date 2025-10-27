static int write_memory(DumpState *s, RAMBlock *block, ram_addr_t start,

                        int64_t size)

{

    int64_t i;

    int ret;



    for (i = 0; i < size / TARGET_PAGE_SIZE; i++) {

        ret = write_data(s, block->host + start + i * TARGET_PAGE_SIZE,

                         TARGET_PAGE_SIZE);

        if (ret < 0) {

            return ret;

        }

    }



    if ((size % TARGET_PAGE_SIZE) != 0) {

        ret = write_data(s, block->host + start + i * TARGET_PAGE_SIZE,

                         size % TARGET_PAGE_SIZE);

        if (ret < 0) {

            return ret;

        }

    }



    return 0;

}
