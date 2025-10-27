static int is_dup_page(uint8_t *page, uint8_t ch)

{

    uint32_t val = ch << 24 | ch << 16 | ch << 8 | ch;

    uint32_t *array = (uint32_t *)page;

    int i;



    for (i = 0; i < (TARGET_PAGE_SIZE / 4); i++) {

        if (array[i] != val)

            return 0;

    }



    return 1;

}
