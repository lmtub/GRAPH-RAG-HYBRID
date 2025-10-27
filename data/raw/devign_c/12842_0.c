static int is_dup_page(uint8_t *page)

{

    VECTYPE *p = (VECTYPE *)page;

    VECTYPE val = SPLAT(page);

    int i;



    for (i = 0; i < TARGET_PAGE_SIZE / sizeof(VECTYPE); i++) {

        if (!ALL_EQ(val, p[i])) {

            return 0;

        }

    }



    return 1;

}
